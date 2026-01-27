#include "rtp_llm/cpp/engine_base/schedulers/SmartScheduler.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <chrono>
#include <memory>
#include <mutex>

using namespace std;
namespace rtp_llm {

void SmartScheduler::accountBatchMetrics(const list<GenerateStreamPtr>& new_streams,
                                         const list<GenerateStreamPtr>& running_streams) {
    size_t total_prefill_len = 0;
    for (auto& stream : new_streams) {
        total_prefill_len += stream->currentExecuteTokenSize();
    }
    for (auto& stream : running_streams) {
        stream->incBatchWithPrefillTimes(new_streams.size());
        stream->incBatchWithPrefillLen(total_prefill_len);
    }
}

bool SmartScheduler::waitPredicate() {
    return stop_ || !waiting_streams_.empty() || !running_streams_.empty() || !remote_running_streams_.empty();
}

absl::StatusOr<std::list<GenerateStreamPtr>> SmartScheduler::schedule(size_t reserve_step) {
    unique_lock<std::mutex> lock(lock_);

    if (waiting_streams_.empty() && running_streams_.empty() && remote_running_streams_.empty()) {
        RTP_LLM_LOG_INFO("SmartScheduler is idle. Resetting finished_parents_ statistics for the next batch.");
        printf("SmartScheduler is idle. Resetting finished_parents_ statistics for the next batch.\n");
        printf("max_generate_batch_size %ld\n", max_generate_batch_size_);
        finished_parents_.clear();
    }

    if (need_fill_fake_stream_) {
        cond_.wait_for(lock, std::chrono::milliseconds(10), [this] { return waitPredicate(); });
    } else {
        cond_.wait(lock, [this] { return waitPredicate(); });
    }

    evaluateRunningRemote();
    if (finished_parents_.size() <= max_generate_batch_size_) {
        updateParentStatus(running_streams_);
        reorderWaitingQueueByLJF();
    }

    evictDoneStreams(waiting_streams_);
    evictDoneStreams(running_streams_);
    evictDoneStreams(remote_running_streams_);
    auto [fallback_streams, error_streams] = evaluateRunningNext(reserve_step);
    auto new_streams                       = scheduleNew(reserve_step);

    // for(auto new_stream : new_streams){
    //     auto it = finished_parents_.find(new_stream->ParentRequestId());
    //     size_t output_len=2049;
    //     if(it != finished_parents_.end()){
    //         output_len=it->second;
    //     }

    //     printf("new_stream add waiting %ld, outputlen %ld \n", new_stream->ParentRequestId(), output_len);
    // }
    accountBatchMetrics(new_streams, running_streams_);
    running_streams_.insert(running_streams_.end(), new_streams.begin(), new_streams.end());
    reportMetrics(fallback_streams);
    last_schedule_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    return running_streams_;
}

void SmartScheduler::updateParentStatus(const std::list<GenerateStreamPtr>& streams) {
    for (const auto& stream : streams) {
        if (stream->stopped() || stream->finished()) {
            int64_t parent_id = stream->ParentRequestId();

            if (finished_parents_.find(parent_id) == finished_parents_.end()) {
                size_t outputlen             = stream->outputTokenLen();
                finished_parents_[parent_id] = outputlen;
            }
        }
    }
}

void SmartScheduler::reorderWaitingQueueByLJF() {
    if (waiting_streams_.size() < 2)
        return;

    // 启发式排序：
    waiting_streams_.sort([this](const GenerateStreamPtr& a, const GenerateStreamPtr& b) {
        auto it_a = finished_parents_.find(a->ParentRequestId());
        auto it_b = finished_parents_.find(b->ParentRequestId());

        bool a_finished = (it_a != finished_parents_.end());
        bool b_finished = (it_b != finished_parents_.end());

        if (a_finished != b_finished) {
            return a_finished;  // 如果 a 没完成，则 a 排在前面
        }

        // 优先级 2: 如果都完成过，按照 token 数从多到少排序（长作业优先）
        if (a_finished && b_finished) {
            if (it_a->second != it_b->second) {
                return it_a->second > it_b->second;  // Token 数多的在前
            }
        }
        // 优先级 3: 兜底逻辑，按流本身的 ID 顺序（FIFO）
        return a->streamId() < b->streamId();
    });
}

void SmartScheduler::reorderWaitingQueueBySJF() {
    if (waiting_streams_.size() < 2)
        return;

    // SJF: Shortest Job First (短作业优先)
    waiting_streams_.sort([this](const GenerateStreamPtr& a, const GenerateStreamPtr& b) {
        auto it_a = finished_parents_.find(a->ParentRequestId());
        auto it_b = finished_parents_.find(b->ParentRequestId());

        bool a_has_history = (it_a != finished_parents_.end());
        bool b_has_history = (it_b != finished_parents_.end());

        if (a_has_history != b_has_history)
            return !a_has_history;

        if (a_has_history && b_has_history) {
            if (it_a->second != it_b->second) {
                return it_a->second < it_b->second;  // Token 数少的排在前面
            }
        }
        return a->streamId() < b->streamId();
    });
}

int64_t SmartScheduler::lastScheduleTime() {
    return empty() ? autil::TimeUtility::currentTimeInMilliSeconds() : last_schedule_time_.load();
}

void SmartScheduler::evictDoneStreams(list<GenerateStreamPtr>& streams) {
    for (auto it = streams.begin(); it != streams.end();) {
        (*it)->checkTimeout();
        if ((*it)->stopped() || (*it)->finished()) {
            // Immediately free resources to run more streams
            (*it)->releaseResource();
            RTP_LLM_LOG_DEBUG("evict stream [%ld]", (*it)->streamId());
            it = streams.erase(it);
        } else {
            ++it;
        }
    }
}

void SmartScheduler::evaluateRunningRemote() {
    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        if ((*it)->needRemoteGenerate() && (*it)->setRemoteGenerate()) {
            remote_running_streams_.emplace_back(*it);
            RTP_LLM_LOG_DEBUG("stream [%ld] move to remote running streams", (*it)->streamId());
            it = running_streams_.erase(it);
        } else {
            ++it;
        }
    }
}

tuple<int, int> SmartScheduler::evaluateRunningNext(size_t reserve_step) {
    // Only in the case of partial fallback, the stream in the waiting queue may hold blocks resources.
    int fallback_streams = 0;
    int error_streams    = 0;

    if (enable_partial_fallback_) {
        for (auto& stream : waiting_streams_) {
            int need_block_num = (int)runningNextBlockNum(reserve_step) - (int)cache_manager_->availableBlockNums();
            if (need_block_num <= 0) {
                break;
            }
            if (stream->maxBlockSize()) {
                RTP_LLM_LOG_INFO("lack mem, stream [%ld] in watting queue try release blocks, "
                                 "it's input_length:%d seq_length:%d, hold block size:%d, release block size:%d",
                                 stream->streamId(),
                                 stream->inputLength(),
                                 stream->seqLength(),
                                 stream->maxBlockSize(),
                                 need_block_num);
                stream->tryReleaseKVBlock(need_block_num);

                if (stream->spIterCount() > 0 || stream->hasNumBeams()) {
                    // sp and beam search do not support partial fallback
                    stream->releaseResource();
                    stream->setStop(ErrorCode::MALLOC_FAILED, "cancel stream since lack kv memory");
                }
                fallback_streams++;
            }
        }
    }

    if (enable_whole_fallback_) {
        while (!running_streams_.empty()) {
            int need_block_num = (int)runningNextBlockNum(reserve_step) - (int)cache_manager_->availableBlockNums();
            if (need_block_num <= 0) {
                break;
            }
            auto& last_stream         = *(running_streams_.rbegin());
            int   need_release_blocks = enable_partial_fallback_ ? need_block_num : last_stream->maxBlockSize();
            RTP_LLM_LOG_INFO(
                "lack mem, stream [%ld] fallback to wait, it's input_length:%d seq_length:%d, hold block size:%d, release block size:%d",
                last_stream->streamId(),
                last_stream->inputLength(),
                last_stream->seqLength(),
                last_stream->maxBlockSize(),
                need_release_blocks);
            last_stream->tryReleaseKVBlock(need_release_blocks);
            if (last_stream->spIterCount() > 0) {
                // sp doesn't support fallback
                last_stream->releaseResource();
                last_stream->setStop(ErrorCode::MALLOC_FAILED, "cancel stream since lack kv memory");
            } else {
                last_stream->setPaused();
                waiting_streams_.emplace_front(last_stream);
            }
            running_streams_.pop_back();
            fallback_streams++;
        }
    }

    if (enable_fast_gen_) {
        token_capacity_ = fast_gen_max_context_len_;
        RTP_LLM_LOG_DEBUG("initial token_capacity is %d", token_capacity_);
    }

    for (auto it = running_streams_.begin(); it != running_streams_.end();) {
        auto result = (*it)->incrKVBlock(token_capacity_, reserve_step);
        if (!result.ok()) {
            (*it)->stopAndRelease(ErrorCode::MALLOC_FAILED, "incrKVBlock failed");
            RTP_LLM_LOG_WARNING("stream [%ld] incr block failed", (*it)->streamId());
            it = running_streams_.erase(it);
            error_streams++;
        } else {
            if (enable_fast_gen_) {
                token_capacity_ -= result.value();
                RTP_LLM_LOG_DEBUG(
                    "after stream [%ld] acquireCapacity, token_capacity is %d", (*it)->streamId(), token_capacity_);
            }
            it++;
        }
    }
    return {fallback_streams, error_streams};
}

bool SmartScheduler::evaluateRunningMemory(const list<GenerateStreamPtr>& streams,
                                           const GenerateStreamPtr&       new_stream) const {
    if (params_.role_type_ == RoleType::DECODE) {
        if (running_streams_.size() + streams.size() + 1 < max_generate_batch_size_) {
            return true;
        }
    }
    if (params_.model_specific_config.load_python_model) {
        // new model py not support prefill and decode togather now
        if (!running_streams_.empty()) {
            return false;
        }
    }
    if (running_streams_.size() + streams.size() + 1 > max_generate_batch_size_) {
        return false;
    }

    if (!enable_fast_gen_) {
        int max_token_size = new_stream->contextLength();
        if (streams.empty() && max_token_size + running_streams_.size() < int(max_seq_len_)) {
            return true;
        }
        for (auto& stream : streams) {
            max_token_size = std::max(max_token_size, stream->contextLength());
        }
        return max_token_size * (streams.size() + 1) + running_streams_.size() < int(max_batch_tokens_size_);
    } else {
        return true;
    }
}

int SmartScheduler::runningNextBlockNum(size_t reserve_step) const {
    int total_need_block_nums = 0;
    for (auto& stream : running_streams_) {
        total_need_block_nums += stream->nextNeedBlockNums(reserve_step);
    }
    return total_need_block_nums;
}

}  // namespace rtp_llm