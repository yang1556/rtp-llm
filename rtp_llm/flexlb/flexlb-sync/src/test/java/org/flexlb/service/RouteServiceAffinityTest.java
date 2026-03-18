package org.flexlb.service;

import org.flexlb.balance.affinity.AffinityAwareRejectionPolicy;
import org.flexlb.balance.affinity.ChatIdAffinityService;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceAffinityTest {

    @Mock
    private ConfigService configService;
    @Mock
    private DefaultRouter defaultRouter;
    @Mock
    private QueueManager queueManager;
    @Mock
    private AffinityAwareRejectionPolicy rejectionPolicy;
    @Mock
    private ResourceMeasureFactory resourceMeasureFactory;
    @Mock
    private ChatIdAffinityService chatIdAffinityService;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        routeService = new RouteService(configService, defaultRouter, queueManager,
                rejectionPolicy, resourceMeasureFactory, chatIdAffinityService);
        // Clear static worker status for test isolation
        clearWorkerStatus();
    }

    @AfterEach
    void tearDown() {
        clearWorkerStatus();
    }

    private void clearWorkerStatus() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    private FlexlbConfig directRoutingConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableQueueing(false);
        config.setEnableChatIdAffinity(true);
        config.setAffinityRejectionWaterLevelThreshold(70.0);
        return config;
    }

    private BalanceContext createContext(String chatId) {
        Request request = new Request();
        request.setChatId(chatId);
        request.setRequestId(1L);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }

    private Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    @Nested
    class WaterLevelBelowThreshold {

        @Test
        void normalRouting_noRejection() {
            FlexlbConfig config = directRoutingConfig();
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(rejectionPolicy.shouldRejectByWaterLevel(any(), anyDouble(), anyDouble())).thenReturn(false);
            when(defaultRouter.route(any())).thenReturn(successResponse());

            BalanceContext ctx = createContext("chat-1");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertTrue(response.isSuccess());
        }
    }

    @Nested
    class WaterLevelAboveThreshold {

        @Test
        void noAffinity_rejected() {
            FlexlbConfig config = directRoutingConfig();
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(rejectionPolicy.shouldRejectByWaterLevel(any(), anyDouble(), anyDouble())).thenReturn(true);

            BalanceContext ctx = createContext("chat-new");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
            verify(defaultRouter, never()).route(any());
        }

        @Test
        void hasAffinity_routeSucceeds() {
            FlexlbConfig config = directRoutingConfig();
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(rejectionPolicy.shouldRejectByWaterLevel(any(), anyDouble(), anyDouble())).thenReturn(false);
            when(defaultRouter.route(any())).thenReturn(successResponse());

            BalanceContext ctx = createContext("chat-existing");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertTrue(response.isSuccess());
        }
    }

    @Nested
    class WaterLevel100 {

        @Test
        void rejectsAll() {
            FlexlbConfig config = directRoutingConfig();
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(rejectionPolicy.shouldRejectByWaterLevel(any(), anyDouble(), anyDouble())).thenReturn(true);

            BalanceContext ctx = createContext("chat-existing");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        }
    }

    @Nested
    class CalculateWaterLevel {

        @Test
        void noWorkers_returnsZeroWaterLevel() {
            FlexlbConfig config = directRoutingConfig();
            when(configService.loadBalanceConfig()).thenReturn(config);
            // No workers → water level = 0 → below threshold → no rejection
            when(rejectionPolicy.shouldRejectByWaterLevel(any(), anyDouble(), anyDouble())).thenReturn(false);
            when(defaultRouter.route(any())).thenReturn(successResponse());

            BalanceContext ctx = createContext("chat-1");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertTrue(response.isSuccess());
        }

        @Test
        void multipleRoles_takesMaxWaterLevel() {
            FlexlbConfig config = directRoutingConfig();
            when(configService.loadBalanceConfig()).thenReturn(config);

            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().put("decode1", new WorkerStatus());
            EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().put("prefill1", new WorkerStatus());

            ResourceMeasure decodeMeasure = mock(ResourceMeasure.class);
            ResourceMeasure prefillMeasure = mock(ResourceMeasure.class);
            when(decodeMeasure.calculateAverageWaterLevel(any())).thenReturn(80.0);
            when(prefillMeasure.calculateAverageWaterLevel(any())).thenReturn(50.0);
            when(resourceMeasureFactory.getMeasure(ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE)).thenReturn(decodeMeasure);
            when(resourceMeasureFactory.getMeasure(ResourceMeasureIndicatorEnum.WAIT_TIME)).thenReturn(prefillMeasure);

            // Max water level is 80.0 (decode) → above threshold → rejection for non-affinity
            when(rejectionPolicy.shouldRejectByWaterLevel(any(), anyDouble(), anyDouble())).thenReturn(true);

            BalanceContext ctx = createContext("chat-new");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        }
    }

    @Nested
    class QueueModeUnchanged {

        @Test
        void queueMode_delegatesToQueueManager() {
            FlexlbConfig config = new FlexlbConfig();
            config.setEnableQueueing(true);
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(queueManager.tryRouteAsync(any())).thenReturn(Mono.just(successResponse()));

            BalanceContext ctx = createContext("chat-1");
            Response response = routeService.route(ctx).block();

            assertNotNull(response);
            assertTrue(response.isSuccess());
            verify(queueManager).tryRouteAsync(any());
            verify(defaultRouter, never()).route(any());
        }
    }
}
