package org.flexlb.balance.scheduler;

import org.flexlb.balance.affinity.AffinityAwareRejectionPolicy;
import org.flexlb.balance.affinity.AffinityAwareRejectionPolicy.RejectionReason;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class QueueManagerAffinityTest {

    @Mock
    private RoutingQueueReporter metrics;

    @Mock
    private ConfigService configService;

    @Mock
    private AffinityAwareRejectionPolicy rejectionPolicy;

    private QueueManager queueManager;

    private static final int SMALL_QUEUE_SIZE = 2;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxQueueSize(SMALL_QUEUE_SIZE);
        when(configService.loadBalanceConfig()).thenReturn(config);
        queueManager = new QueueManager(metrics, configService, rejectionPolicy);
    }

    private BalanceContext createCtx(String chatId) {
        Request request = new Request();
        request.setChatId(chatId);
        request.setGenerateTimeout(60_000);
        request.setRequestId(System.nanoTime());

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }

    /**
     * Queue full + no affinity → reject with QUEUE_FULL
     */
    @Test
    void queueFull_noAffinity_rejectsWithQueueFull() {
        // Fill the queue
        for (int i = 0; i < SMALL_QUEUE_SIZE; i++) {
            BalanceContext filler = createCtx("filler-" + i);
            queueManager.tryRouteAsync(filler);
        }

        // New request with no affinity
        when(rejectionPolicy.shouldRejectRequest(any(Request.class), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(true);

        BalanceContext newCtx = createCtx("no-affinity-chat");
        Response response = queueManager.tryRouteAsync(newCtx).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
        verify(metrics, atLeastOnce()).reportRejected();
    }

    /**
     * Queue full + has affinity + replaceable non-affinity in queue → replace and enqueue
     */
    @Test
    void queueFull_hasAffinity_replacesNonAffinityRequest() {
        // Fill the queue with non-affinity requests
        BalanceContext filler1 = createCtx("no-affinity-1");
        BalanceContext filler2 = createCtx("no-affinity-2");
        queueManager.tryRouteAsync(filler1);
        queueManager.tryRouteAsync(filler2);

        BalanceContext affinityCtx = createCtx("affinity-chat");

        // For the new request: don't reject (has affinity)
        when(rejectionPolicy.shouldRejectRequest(eq(affinityCtx.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(false);
        // Tail item (filler2) is replaceable - iteration goes from tail
        when(rejectionPolicy.shouldRejectRequest(eq(filler2.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(true);

        // Don't block - the Mono wraps a future that won't complete without a scheduler
        queueManager.tryRouteAsync(affinityCtx);

        // filler2 (tail) should be replaced, its future completed with QUEUE_FULL
        assertTrue(filler2.getFuture().isDone(),
                "Tail non-affinity request should have been replaced");

        // The affinity request's future should still be pending (enqueued, not rejected)
        assertFalse(affinityCtx.getFuture().isDone(),
                "Affinity request should be enqueued, not immediately completed");
    }

    /**
     * Queue full + has affinity + no replaceable request → reject with QUEUE_FULL
     */
    @Test
    void queueFull_hasAffinity_noReplaceable_rejectsWithQueueFull() {
        // Fill the queue with affinity requests (not replaceable)
        BalanceContext filler1 = createCtx("affinity-1");
        BalanceContext filler2 = createCtx("affinity-2");
        queueManager.tryRouteAsync(filler1);
        queueManager.tryRouteAsync(filler2);

        BalanceContext newCtx = createCtx("new-affinity-chat");

        // New request has affinity (don't reject)
        when(rejectionPolicy.shouldRejectRequest(eq(newCtx.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(false);
        // Queue items also have affinity (not replaceable)
        when(rejectionPolicy.shouldRejectRequest(eq(filler1.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(false);
        when(rejectionPolicy.shouldRejectRequest(eq(filler2.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(false);

        Response response = queueManager.tryRouteAsync(newCtx).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
        verify(metrics, atLeastOnce()).reportRejected();
    }

    /**
     * Feature disabled → original behavior (always reject on queue full)
     */
    @Test
    void featureDisabled_alwaysRejectsOnQueueFull() {
        // Fill the queue
        for (int i = 0; i < SMALL_QUEUE_SIZE; i++) {
            BalanceContext filler = createCtx("filler-" + i);
            queueManager.tryRouteAsync(filler);
        }

        // Feature disabled: shouldRejectRequest always returns true
        when(rejectionPolicy.shouldRejectRequest(any(Request.class), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(true);

        BalanceContext newCtx = createCtx("any-chat");
        Response response = queueManager.tryRouteAsync(newCtx).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
        verify(metrics, atLeastOnce()).reportRejected();
    }

    /**
     * Queue not full → request is enqueued normally regardless of affinity
     */
    @Test
    void queueNotFull_requestEnqueuedNormally() {
        BalanceContext ctx = createCtx("any-chat");
        queueManager.tryRouteAsync(ctx);

        // Request should be enqueued, future should not be completed yet
        assertNotNull(ctx.getFuture());
        assertFalse(ctx.getFuture().isDone());
        verify(metrics).reportQueueEntry();
        // rejectionPolicy should NOT be called when queue is not full
        verify(rejectionPolicy, never()).shouldRejectRequest(any(), any());
    }

    /**
     * Replaced request's future is completed with QUEUE_FULL error
     */
    @Test
    void replacedRequest_futureCompletedWithQueueFull() {
        // Fill queue with non-affinity requests
        BalanceContext filler1 = createCtx("no-affinity-1");
        BalanceContext filler2 = createCtx("no-affinity-2");
        queueManager.tryRouteAsync(filler1);
        queueManager.tryRouteAsync(filler2);

        BalanceContext affinityCtx = createCtx("affinity-chat");

        when(rejectionPolicy.shouldRejectRequest(eq(affinityCtx.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(false);
        // Tail item (filler2) is replaceable
        when(rejectionPolicy.shouldRejectRequest(eq(filler2.getRequest()), eq(RejectionReason.QUEUE_FULL)))
                .thenReturn(true);

        queueManager.tryRouteAsync(affinityCtx);

        // filler2 (tail) should be replaced
        assertTrue(filler2.getFuture().isDone());

        Response replacedResponse = filler2.getFuture().join();
        assertFalse(replacedResponse.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), replacedResponse.getCode());
    }
}
