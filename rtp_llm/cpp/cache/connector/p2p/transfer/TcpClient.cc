#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"

#include "aios/network/arpc/arpc/metric/KMonitorANetClientMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace transfer {

TcpClient::~TcpClient() {
    stop();
}

bool TcpClient::init(int io_thread_count) {
    if (rpc_channel_transport_ == nullptr) {
        rpc_channel_transport_.reset(new anet::Transport(io_thread_count));
        if (!rpc_channel_transport_ || !rpc_channel_transport_->start()) {
            stop();
            return false;
        }
        rpc_channel_transport_->setName("MessagerClientRPCChannel");
    }

    rpc_channel_manager_.reset(new arpc::ANetRPCChannelManager(rpc_channel_transport_.get()));
    {
        arpc::KMonitorANetMetricReporterConfig metricConfig;
        metricConfig.arpcConfig.enableArpcMetric = true;
        metricConfig.anetConfig.enableANetMetric = false;
        metricConfig.metricLevel                 = kmonitor::FATAL;
        auto metricReporter = std::make_shared<arpc::KMonitorANetClientMetricReporter>(metricConfig);
        if (!metricReporter->init(rpc_channel_transport_.get())) {
            RTP_LLM_LOG_ERROR("anet metric reporter init failed");
            stop();
            return false;
        }
        rpc_channel_manager_->SetMetricReporter(metricReporter);
    }
    RTP_LLM_LOG_INFO("tcp client init success, io thread count %d", io_thread_count);
    return true;
}

void TcpClient::stop() {
    if (rpc_channel_manager_) {
        rpc_channel_transport_->stop();
        rpc_channel_transport_->wait();

        rpc_channel_manager_->Close();
        rpc_channel_manager_.reset();

        rpc_channel_transport_.reset();
    } else if (rpc_channel_transport_) {
        rpc_channel_transport_->stop();
        rpc_channel_transport_->wait();
        rpc_channel_transport_.reset();
    }
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::getChannel(const std::string& ip, uint32_t port) {
    std::string spec = "tcp:" + ip + ":" + std::to_string(port);

    std::lock_guard<std::mutex> lock(channel_map_mutex_);
    auto                        it = channel_map_.find(spec);
    if (it != channel_map_.end()) {
        const auto& channel = it->second;
        if (channel != nullptr && !channel->ChannelBroken()) {
            return channel;
        }
        // Drop broken or null entries so we never cache unusable channels.
        channel_map_.erase(it);
    }

    auto new_channel = openChannel(spec);
    if (new_channel == nullptr || new_channel->ChannelBroken()) {
        return nullptr;
    }

    channel_map_.emplace(spec, new_channel);
    RTP_LLM_LOG_INFO("tcp client new channel connect to %s", spec.c_str());
    return new_channel;
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::openChannel(const std::string& spec) {
    if (!rpc_channel_manager_) {
        RTP_LLM_LOG_WARNING("tcp client open channel to %s failed, rpc channel manager is null", spec.c_str());
        return nullptr;
    }

    return std::shared_ptr<arpc::RPCChannelBase>(
        dynamic_cast<arpc::RPCChannelBase*>(rpc_channel_manager_->OpenChannel(spec, false, 1000ul)));
}

}  // namespace transfer
}  // namespace rtp_llm
