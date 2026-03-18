package org.flexlb.balance.affinity;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.springframework.stereotype.Component;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Affinity-aware rejection policy that decides whether to reject a request based on
 * affinity information and water level. In queue mode, rejection is based on queue-full
 * status; in direct routing mode, progressive rejection is based on cluster water level.
 */
@Slf4j
@Component
public class AffinityAwareRejectionPolicy {

    private final ChatIdAffinityService affinityService;
    private final ConfigService configService;

    /**
     * Rejection scenario enum
     */
    public enum RejectionReason {
        QUEUE_FULL,           // Queue mode: queue is full
        NO_AVAILABLE_WORKER   // Direct routing mode: all workers busy (water level 100%)
    }

    public AffinityAwareRejectionPolicy(ChatIdAffinityService affinityService, ConfigService configService) {
        this.affinityService = affinityService;
        this.configService = configService;
    }

    /**
     * Determines whether a request should be rejected (used in queue mode).
     *
     * @param request the request to evaluate
     * @param reason  rejection reason (queue full / worker unavailable)
     * @return true if the request should be rejected (no affinity)
     */
    public boolean shouldRejectRequest(Request request, RejectionReason reason) {
        FlexlbConfig config = configService.loadBalanceConfig();

        // If affinity feature is disabled, use default rejection policy
        if (!config.isEnableChatIdAffinity()) {
            return true;
        }

        // Check if request has a chatId
        String chatId = request.getChatId();
        if (chatId == null || chatId.isEmpty()) {
            return true;
        }

        // Query affinity: reject requests without affinity first, try to preserve those with affinity
        return !affinityService.hasAffinity(chatId);
    }

    /**
     * Water-level-based progressive rejection (used in direct routing mode).
     *
     * <p>Decision logic (mirrors the smooth flow control in DynamicWorkerManager):
     * <ul>
     *   <li>waterLevel &lt; threshold → accept all requests (return false)</li>
     *   <li>waterLevel &gt;= 100 → reject all requests (return true)</li>
     *   <li>threshold &lt;= waterLevel &lt; 100, no affinity →
     *       probabilistic rejection: rejectionRate = (waterLevel - threshold) / (100 - threshold)</li>
     *   <li>threshold &lt;= waterLevel &lt; 100, has affinity → accept (return false)</li>
     * </ul>
     *
     * @param request    the request to evaluate
     * @param waterLevel current cluster water level (0-100), calculated by ResourceMeasure
     * @param threshold  water level rejection threshold (default 70), from config affinityRejectionWaterLevelThreshold
     * @return true if the request should be rejected
     */
    public boolean shouldRejectByWaterLevel(Request request, double waterLevel, double threshold) {
        FlexlbConfig config = configService.loadBalanceConfig();

        // If affinity feature is disabled, skip water level rejection (preserve original behavior)
        if (!config.isEnableChatIdAffinity()) {
            return false;
        }

        // Water level below threshold, accept all requests normally
        if (waterLevel < threshold) {
            return false;
        }

        // Water level at 100%, reject all requests (regardless of affinity)
        if (waterLevel >= 100.0) {
            return true;
        }

        // Water level in [threshold, 100%) range, enter "smooth rejection zone"
        // Requests with affinity always pass in this zone
        String chatId = request.getChatId();
        if (chatId != null && !chatId.isEmpty() && affinityService.hasAffinity(chatId)) {
            return false;
        }

        // No affinity: probabilistic rejection, linearly increasing with water level
        // At threshold → 0% rejection, at 100% → 100% rejection
        double rejectionRate = (waterLevel - threshold) / (100.0 - threshold);
        return ThreadLocalRandom.current().nextDouble() < rejectionRate;
    }
}
