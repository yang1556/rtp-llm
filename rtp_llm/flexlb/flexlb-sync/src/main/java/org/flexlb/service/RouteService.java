package org.flexlb.service;

import java.util.concurrent.CancellationException;

import org.flexlb.balance.affinity.AffinityAwareRejectionPolicy;
import org.flexlb.balance.affinity.ChatIdAffinityService;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final AffinityAwareRejectionPolicy rejectionPolicy;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final ChatIdAffinityService chatIdAffinityService;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        AffinityAwareRejectionPolicy rejectionPolicy,
                        ResourceMeasureFactory resourceMeasureFactory,
                        ChatIdAffinityService chatIdAffinityService) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.rejectionPolicy = rejectionPolicy;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.chatIdAffinityService = chatIdAffinityService;
    }

    /**
     * Route request to appropriate workers
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public Mono<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        Mono<Response> resultMono;
        if (flexlbConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // Use async queuing mechanism
        } else {
            // Direct routing with water level rejection
            resultMono = routeWithWaterLevelCheck(balanceContext, flexlbConfig);
        }

        return resultMono.doOnSuccess(result -> {
            balanceContext.setResponse(result);
        });
    }

    /**
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        if (flexlbConfig.isEnableQueueing()) {
            balanceContext.cancel();
            balanceContext.getFuture().completeExceptionally(new CancellationException("Request cancelled by client"));
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }

    /**
     * Direct routing with water level check.
     * Skips water level calculation entirely when affinity feature is disabled.
     * When enabled, calculates cluster water level and rejects non-affinity requests above threshold.
     */
    private Mono<Response> routeWithWaterLevelCheck(BalanceContext balanceContext, FlexlbConfig flexlbConfig) {
        // Skip expensive water level calculation when affinity feature is disabled
        if (flexlbConfig.isEnableChatIdAffinity()) {
            double waterLevel = resourceMeasureFactory.calculateMaxWaterLevel(flexlbConfig);
            double threshold = flexlbConfig.getAffinityRejectionWaterLevelThreshold();

            if (rejectionPolicy.shouldRejectByWaterLevel(balanceContext.getRequest(), waterLevel, threshold)) {
                return Mono.just(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
            }
        }

        // Normal routing via router
        Response response = router.route(balanceContext);
        return Mono.just(response);
    }


}
