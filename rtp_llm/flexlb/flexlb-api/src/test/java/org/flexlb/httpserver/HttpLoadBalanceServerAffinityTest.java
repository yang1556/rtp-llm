package org.flexlb.httpserver;

import org.flexlb.balance.affinity.ChatIdAffinityService;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.lang.reflect.Method;

import static org.mockito.Mockito.*;

/**
 * Tests for affinity recording integration in HttpLoadBalanceServer.
 * Validates Requirements 8.1 and 8.2.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("HttpLoadBalanceServer Affinity Recording Tests")
class HttpLoadBalanceServerAffinityTest {

    @Mock
    private GeneralHttpNettyService generalHttpNettyService;
    @Mock
    private RouteService routeService;
    @Mock
    private LBStatusConsistencyService lbStatusConsistencyService;
    @Mock
    private EngineHealthReporter engineHealthReporter;
    @Mock
    private QueueManager queueManager;
    @Mock
    private ActiveRequestCounter activeRequestCounter;
    @Mock
    private ChatIdAffinityService chatIdAffinityService;

    private HttpLoadBalanceServer server;

    @BeforeEach
    void setUp() {
        server = new HttpLoadBalanceServer(
                generalHttpNettyService,
                routeService,
                lbStatusConsistencyService,
                engineHealthReporter,
                queueManager,
                activeRequestCounter,
                chatIdAffinityService
        );
    }

    private Mono<ServerResponse> invokeHandleRoutingResult(BalanceContext ctx, Response response) throws Exception {
        Method method = HttpLoadBalanceServer.class.getDeclaredMethod(
                "handleRoutingResult", BalanceContext.class, Response.class);
        method.setAccessible(true);
        @SuppressWarnings("unchecked")
        Mono<ServerResponse> result = (Mono<ServerResponse>) method.invoke(server, ctx, response);
        return result;
    }

    @Test
    @DisplayName("Should record affinity when routing succeeds with valid chatId")
    void handleRoutingResult_shouldRecordAffinity_whenSuccessWithChatId() throws Exception {
        Request request = new Request();
        request.setChatId("chat-123");
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        Response response = new Response();
        response.setSuccess(true);

        Mono<ServerResponse> result = invokeHandleRoutingResult(ctx, response);

        StepVerifier.create(result)
                .expectNextMatches(r -> r.statusCode().value() == 200)
                .verifyComplete();

        verify(chatIdAffinityService, times(1)).recordAffinity("chat-123");
    }

    @Test
    @DisplayName("Should NOT record affinity when routing succeeds but chatId is null")
    void handleRoutingResult_shouldNotRecordAffinity_whenSuccessWithNullChatId() throws Exception {
        Request request = new Request();
        // chatId is null by default
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        Response response = new Response();
        response.setSuccess(true);

        Mono<ServerResponse> result = invokeHandleRoutingResult(ctx, response);

        StepVerifier.create(result)
                .expectNextMatches(r -> r.statusCode().value() == 200)
                .verifyComplete();

        verify(chatIdAffinityService, never()).recordAffinity(any());
    }

    @Test
    @DisplayName("Should NOT record affinity when routing succeeds but chatId is empty")
    void handleRoutingResult_shouldNotRecordAffinity_whenSuccessWithEmptyChatId() throws Exception {
        Request request = new Request();
        request.setChatId("");
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        Response response = new Response();
        response.setSuccess(true);

        Mono<ServerResponse> result = invokeHandleRoutingResult(ctx, response);

        StepVerifier.create(result)
                .expectNextMatches(r -> r.statusCode().value() == 200)
                .verifyComplete();

        verify(chatIdAffinityService, never()).recordAffinity(any());
    }

    @Test
    @DisplayName("Should NOT record affinity when routing fails")
    void handleRoutingResult_shouldNotRecordAffinity_whenRoutingFails() throws Exception {
        Request request = new Request();
        request.setChatId("chat-456");
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        Response response = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);

        Mono<ServerResponse> result = invokeHandleRoutingResult(ctx, response);

        StepVerifier.create(result)
                .expectNextMatches(r -> r.statusCode().value() == 500)
                .verifyComplete();

        verify(chatIdAffinityService, never()).recordAffinity(any());
    }
}
