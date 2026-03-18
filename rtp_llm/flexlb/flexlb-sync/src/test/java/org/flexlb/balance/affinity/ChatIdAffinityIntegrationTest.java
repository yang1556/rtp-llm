package org.flexlb.balance.affinity;

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
import org.flexlb.service.RouteService;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * End-to-end integration tests for Chat ID Affinity feature.
 *
 * Uses real ChatIdAffinityIndex, ChatIdAffinityService, and AffinityAwareRejectionPolicy.
 * Mocks Router and ResourceMeasure for controlling water levels and routing outcomes.
 */
@ExtendWith(MockitoExtension.class)
class ChatIdAffinityIntegrationTest {

    @Mock
    private DefaultRouter defaultRouter;
    @Mock
    private ResourceMeasureFactory resourceMeasureFactory;
    @Mock
    private ResourceMeasure resourceMeasure;
    @Mock
    private RoutingQueueReporter metrics;

    // Real objects — initialized per test
    private ChatIdAffinityService affinityService;
    private AffinityAwareRejectionPolicy rejectionPolicy;

    @BeforeEach
    void setUp() {
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

    private ConfigService mockConfigService(FlexlbConfig config) {
        ConfigService cs = mock(ConfigService.class);
        when(cs.loadBalanceConfig()).thenReturn(config);
        return cs;
    }

    private FlexlbConfig enabledConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        config.setEnableQueueing(false);
        config.setChatIdAffinityExpirationMs(600_000);
        config.setChatIdAffinityMaxEntries(1000);
        config.setAffinityRejectionWaterLevelThreshold(70.0);
        return config;
    }

    private FlexlbConfig disabledConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(false);
        config.setEnableQueueing(false);
        return config;
    }

    private void initRealServices(FlexlbConfig config) {
        ConfigService cs = mockConfigService(config);
        affinityService = new ChatIdAffinityService(cs);
        affinityService.init();
        rejectionPolicy = new AffinityAwareRejectionPolicy(affinityService, cs);
    }

    private BalanceContext createContext(String chatId) {
        Request request = new Request();
        request.setChatId(chatId);
        request.setRequestId(System.nanoTime());
        request.setGenerateTimeout(60_000);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }

    private Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    private void setupWaterLevel(double waterLevel) {
        lenient().when(resourceMeasureFactory.calculateMaxWaterLevel(any())).thenReturn(waterLevel);
    }

    // ========================================================================
    // 8.1 Feature toggle off — direct routing mode
    // ========================================================================
    @Test
    void featureDisabled_directRouting_shouldRejectByWaterLevelReturnsFalse_normalRoutingProceeds() {
        FlexlbConfig config = disabledConfig();
        initRealServices(config);

        setupWaterLevel(80.0);
        when(defaultRouter.route(any())).thenReturn(successResponse());

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        BalanceContext ctx = createContext("chat-123");
        Response response = routeService.route(ctx).block();

        assertNotNull(response);
        assertTrue(response.isSuccess(),
                "With feature disabled, shouldRejectByWaterLevel returns false, so routing proceeds normally");
        verify(defaultRouter).route(any());
    }

    // ========================================================================
    // 8.1 Feature toggle off — queue mode
    // ========================================================================
    @Test
    void featureDisabled_queueMode_shouldRejectRequestReturnsTrue_defaultRejectBehavior() {
        FlexlbConfig config = disabledConfig();
        config.setEnableQueueing(true);
        config.setMaxQueueSize(2);
        initRealServices(config);

        QueueManager queueManager = new QueueManager(metrics, mockConfigService(config), rejectionPolicy);

        // Fill the queue
        queueManager.tryRouteAsync(createContext("filler-1"));
        queueManager.tryRouteAsync(createContext("filler-2"));

        // Feature disabled → shouldRejectRequest returns true (default reject)
        BalanceContext ctx = createContext("any-chat");
        Response response = queueManager.tryRouteAsync(ctx).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
    }

    // ========================================================================
    // 8.2 Routing independence — affinity only affects rejection, not routing
    // ========================================================================
    @Test
    void routingIndependence_routerRouteCallIsSameRegardlessOfAffinity() {
        FlexlbConfig config = enabledConfig();
        initRealServices(config);

        // Water level below threshold → no rejection for either request
        setupWaterLevel(50.0);
        when(defaultRouter.route(any())).thenReturn(successResponse());

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        // Request with affinity
        affinityService.recordAffinity("chat-affinity");
        BalanceContext affinityCtx = createContext("chat-affinity");
        Response affinityResponse = routeService.route(affinityCtx).block();

        // Request without affinity
        BalanceContext noAffinityCtx = createContext("chat-no-affinity");
        Response noAffinityResponse = routeService.route(noAffinityCtx).block();

        // Both succeed — routing decision is the same (based on TTFT, not affinity)
        assertNotNull(affinityResponse);
        assertNotNull(noAffinityResponse);
        assertTrue(affinityResponse.isSuccess());
        assertTrue(noAffinityResponse.isSuccess());

        // router.route() was called for both
        verify(defaultRouter, times(2)).route(any());
    }

    // ========================================================================
    // 8.3 Direct routing — water level below threshold → all accepted
    // ========================================================================
    @Test
    void directRouting_waterLevelBelowThreshold_allRequestsAccepted() {
        FlexlbConfig config = enabledConfig();
        initRealServices(config);

        setupWaterLevel(50.0); // below 70% threshold
        when(defaultRouter.route(any())).thenReturn(successResponse());

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        // Request without affinity — accepted
        Response r1 = routeService.route(createContext("chat-new")).block();
        assertNotNull(r1);
        assertTrue(r1.isSuccess());

        // Request with affinity — also accepted
        affinityService.recordAffinity("chat-existing");
        Response r2 = routeService.route(createContext("chat-existing")).block();
        assertNotNull(r2);
        assertTrue(r2.isSuccess());

        verify(defaultRouter, times(2)).route(any());
    }

    // ========================================================================
    // 8.3 Direct routing — water level near 100%, no affinity → almost always rejected
    // ========================================================================
    @Test
    void directRouting_waterLevelNear100_noAffinity_rejectedWithNoAvailableWorker() {
        FlexlbConfig config = enabledConfig();
        initRealServices(config);

        // Use 99.9% water level for near-deterministic rejection (rejectionRate ≈ 99.7%)
        setupWaterLevel(99.9);

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        Response response = routeService.route(createContext("chat-new")).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
    }

    // ========================================================================
    // 8.3 Direct routing — water level above threshold, has affinity → accepted
    // ========================================================================
    @Test
    void directRouting_waterLevelAboveThreshold_hasAffinity_acceptedAndRoutedNormally() {
        FlexlbConfig config = enabledConfig();
        initRealServices(config);

        setupWaterLevel(80.0); // above 70% threshold
        when(defaultRouter.route(any())).thenReturn(successResponse());

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        affinityService.recordAffinity("chat-existing");
        Response response = routeService.route(createContext("chat-existing")).block();

        assertNotNull(response);
        assertTrue(response.isSuccess());
        verify(defaultRouter).route(any());
    }

    // ========================================================================
    // 8.3 Direct routing — water level 100% → all rejected
    // ========================================================================
    @Test
    void directRouting_waterLevel100_allRequestsRejectedRegardlessOfAffinity() {
        FlexlbConfig config = enabledConfig();
        initRealServices(config);

        setupWaterLevel(100.0);

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        affinityService.recordAffinity("chat-existing");

        // Request with affinity — still rejected at 100%
        Response r1 = routeService.route(createContext("chat-existing")).block();
        assertNotNull(r1);
        assertFalse(r1.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), r1.getCode());

        // Request without affinity — also rejected
        Response r2 = routeService.route(createContext("chat-new")).block();
        assertNotNull(r2);
        assertFalse(r2.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), r2.getCode());

        verify(defaultRouter, never()).route(any());
    }

    // ========================================================================
    // 8.4 Queue mode — queue full, no affinity → rejected with QUEUE_FULL
    // ========================================================================
    @Test
    void queueMode_queueFull_noAffinity_rejectedWithQueueFull() {
        FlexlbConfig config = enabledConfig();
        config.setEnableQueueing(true);
        config.setMaxQueueSize(2);
        initRealServices(config);

        QueueManager queueManager = new QueueManager(metrics, mockConfigService(config), rejectionPolicy);

        // Fill the queue
        queueManager.tryRouteAsync(createContext("filler-1"));
        queueManager.tryRouteAsync(createContext("filler-2"));

        // New request without affinity
        Response response = queueManager.tryRouteAsync(createContext("no-affinity-chat")).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
    }

    // ========================================================================
    // 8.4 Queue mode — queue full, has affinity, replaceable → replaced and enqueued
    // ========================================================================
    @Test
    void queueMode_queueFull_hasAffinity_replaceableNonAffinityInQueue_replacedAndEnqueued() {
        FlexlbConfig config = enabledConfig();
        config.setEnableQueueing(true);
        config.setMaxQueueSize(2);
        initRealServices(config);

        QueueManager queueManager = new QueueManager(metrics, mockConfigService(config), rejectionPolicy);

        // Fill queue with non-affinity requests
        BalanceContext filler1 = createContext("no-affinity-1");
        BalanceContext filler2 = createContext("no-affinity-2");
        queueManager.tryRouteAsync(filler1);
        queueManager.tryRouteAsync(filler2);

        // Record affinity for the new request
        affinityService.recordAffinity("affinity-chat");

        BalanceContext affinityCtx = createContext("affinity-chat");
        queueManager.tryRouteAsync(affinityCtx);

        // Tail non-affinity request (filler2) should be replaced
        assertTrue(filler2.getFuture().isDone(),
                "Tail non-affinity request should have been replaced");
        Response replacedResponse = filler2.getFuture().join();
        assertFalse(replacedResponse.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), replacedResponse.getCode());

        // Affinity request should be enqueued (future still pending)
        assertFalse(affinityCtx.getFuture().isDone(),
                "Affinity request should be enqueued, not immediately completed");
    }

    // ========================================================================
    // 8.4 Queue mode — queue full, has affinity, no replaceable → rejected
    // ========================================================================
    @Test
    void queueMode_queueFull_hasAffinity_noReplaceable_rejectedWithQueueFull() {
        FlexlbConfig config = enabledConfig();
        config.setEnableQueueing(true);
        config.setMaxQueueSize(2);
        initRealServices(config);

        QueueManager queueManager = new QueueManager(metrics, mockConfigService(config), rejectionPolicy);

        // Fill queue with affinity requests (not replaceable)
        affinityService.recordAffinity("affinity-1");
        affinityService.recordAffinity("affinity-2");
        queueManager.tryRouteAsync(createContext("affinity-1"));
        queueManager.tryRouteAsync(createContext("affinity-2"));

        // New affinity request — no replaceable items in queue
        affinityService.recordAffinity("affinity-3");
        Response response = queueManager.tryRouteAsync(createContext("affinity-3")).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
    }

    // ========================================================================
    // 9.1 Closed-loop: route success → handleRoutingResult records affinity
    //     → subsequent request accepted at high water level
    // ========================================================================
    @Test
    void closedLoop_routeSuccess_recordsAffinity_subsequentRequestAcceptedAtHighWaterLevel() throws Exception {
        FlexlbConfig config = enabledConfig();
        initRealServices(config);

        // Water level above threshold
        setupWaterLevel(80.0);
        when(defaultRouter.route(any())).thenReturn(successResponse());

        RouteService routeService = new RouteService(
                mockConfigService(config), defaultRouter, mock(QueueManager.class),
                rejectionPolicy, resourceMeasureFactory, affinityService);

        // Verify chatId has NO affinity initially
        assertFalse(affinityService.hasAffinity("chat-closed-loop"),
                "chatId should have no affinity before first route");

        // First request: water level below threshold so it goes through
        setupWaterLevel(50.0);
        BalanceContext firstCtx = createContext("chat-closed-loop");
        Response firstResponse = routeService.route(firstCtx).block();
        assertNotNull(firstResponse);
        assertTrue(firstResponse.isSuccess());

        // Simulate handleRoutingResult recording affinity (as HttpLoadBalanceServer does)
        // This is the key: in production, HttpLoadBalanceServer.handleRoutingResult calls recordAffinity
        String chatId = firstCtx.getRequest().getChatId();
        if (chatId != null && !chatId.isEmpty() && firstResponse.isSuccess()) {
            affinityService.recordAffinity(chatId);
        }

        // Verify affinity was recorded
        assertTrue(affinityService.hasAffinity("chat-closed-loop"),
                "chatId should have affinity after successful route + record");

        // Second request: water level above threshold, but has affinity → accepted
        setupWaterLevel(80.0);
        BalanceContext secondCtx = createContext("chat-closed-loop");
        Response secondResponse = routeService.route(secondCtx).block();

        assertNotNull(secondResponse);
        assertTrue(secondResponse.isSuccess(),
                "Request with affinity should be accepted even at high water level");

        // A new chatId without affinity should be rejected at very high water level
        setupWaterLevel(99.9);
        BalanceContext newCtx = createContext("chat-no-history");
        Response newResponse = routeService.route(newCtx).block();
        assertNotNull(newResponse);
        assertFalse(newResponse.isSuccess(),
                "Request without affinity should be rejected at very high water level");
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), newResponse.getCode());
    }

    // ========================================================================
    // 9.4 Queue mode closed-loop: enqueue → complete future with success
    //     → verify affinity recording path works end-to-end
    // ========================================================================
    @Test
    void queueMode_closedLoop_enqueueAndSucceed_affinityRecordedViaHandleRoutingResult() throws Exception {
        FlexlbConfig config = enabledConfig();
        config.setEnableQueueing(true);
        config.setMaxQueueSize(10);
        initRealServices(config);

        QueueManager queueManager = new QueueManager(metrics, mockConfigService(config), rejectionPolicy);

        // Enqueue a request
        BalanceContext ctx = createContext("chat-queue-loop");
        queueManager.tryRouteAsync(ctx);

        // Verify not yet completed
        assertFalse(ctx.getFuture().isDone(), "Request should be waiting in queue");

        // Simulate RequestScheduler dequeuing and routing successfully
        BalanceContext dequeued = queueManager.takeRequest(false, 0);
        assertNotNull(dequeued, "Should dequeue the request");
        assertEquals("chat-queue-loop", dequeued.getRequest().getChatId());

        // Simulate successful routing result
        Response successResp = successResponse();
        dequeued.getFuture().complete(successResp);

        // Verify future completed with success
        Response result = ctx.getFuture().get();
        assertTrue(result.isSuccess());

        // Simulate handleRoutingResult recording affinity (as HttpLoadBalanceServer does)
        String chatId = dequeued.getRequest().getChatId();
        if (chatId != null && !chatId.isEmpty() && result.isSuccess()) {
            affinityService.recordAffinity(chatId);
        }

        // Verify affinity was recorded
        assertTrue(affinityService.hasAffinity("chat-queue-loop"),
                "Affinity should be recorded after queue mode success");

        // Now verify: at high water level in direct mode, this chatId would be accepted
        // (Switch to direct routing to verify the affinity effect)
        FlexlbConfig directConfig = enabledConfig();
        directConfig.setEnableQueueing(false);
        // Re-init rejection policy with same affinity service (affinity data persists)
        AffinityAwareRejectionPolicy directPolicy = new AffinityAwareRejectionPolicy(
                affinityService, mockConfigService(directConfig));

        setupWaterLevel(99.9);
        when(defaultRouter.route(any())).thenReturn(successResponse());

        RouteService directRouteService = new RouteService(
                mockConfigService(directConfig), defaultRouter, mock(QueueManager.class),
                directPolicy, resourceMeasureFactory, affinityService);

        // Request with recorded affinity → accepted at very high water level
        Response affinityResponse = directRouteService.route(createContext("chat-queue-loop")).block();
        assertNotNull(affinityResponse);
        assertTrue(affinityResponse.isSuccess(),
                "chatId with affinity from queue mode should be accepted in direct mode at high water level");

        // Request without affinity → rejected at near-100% water level
        Response noAffinityResponse = directRouteService.route(createContext("chat-unknown")).block();
        assertNotNull(noAffinityResponse);
        assertFalse(noAffinityResponse.isSuccess(),
                "chatId without affinity should be rejected at very high water level");
    }
}
