#pragma once

#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include <set>
#include <list>
#include <mutex>

namespace rtp_llm {

/**
 * @brief SmartScheduler 继承自 FIFOScheduler
 * 专门针对“相同 Prompt 多份副本”场景。通过识别第一个完成的副本（探测请求）来预估任务长度，
 * 并动态提升该 Parent 下其他副本的优先级，从而实现启发式的长/短作业优先（SJF）。
 */
class SmartScheduler: virtual public FIFOScheduler {
public:
    explicit SmartScheduler(const rtp_llm::GptInitParameter&     params,
                            const std::shared_ptr<CacheManager>& cache_manager,
                            const kmonitor::MetricsReporterPtr   metrics_reporter = nullptr,
                            const int                            max_score_len    = 1):
        FIFOScheduler(params, cache_manager, metrics_reporter, max_score_len) {}

    absl::StatusOr<std::list<GenerateStreamPtr>> schedule(size_t reserve_step) override;

private:
    void                 evictDoneStreams(std::list<GenerateStreamPtr>& streams);
    std::tuple<int, int> evaluateRunningNext(size_t reserve_step);
    void                 evaluateRunningRemote();
    int64_t              lastScheduleTime() override;
    int                  runningNextBlockNum(size_t reserve_step) const;
    bool evaluateRunningMemory(const std::list<GenerateStreamPtr>& streams, const GenerateStreamPtr& new_stream) const;
    void accountBatchMetrics(const std::list<GenerateStreamPtr>& new_streams,
                             const std::list<GenerateStreamPtr>& running_streams);
    bool waitPredicate();

protected:
    /**
     * @brief 扫描流列表，更新已经完成探测的 Parent 状态
     */
    void updateParentStatus(const std::list<GenerateStreamPtr>& streams);

    /**
     * @brief 对等待队列进行重排，将已知“短任务”的副本置顶
     */
    void reorderWaitingQueue();

private:
    // 存储已经完成过至少一个副本生成的 parent_id
    std::unordered_map<int64_t, int> finished_parents_;
};

}  // namespace rtp_llm