package org.flexlb.balance.affinity;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ChatIdAffinityServiceTest {

    @Mock
    private ConfigService configService;

    // --- Feature enabled ---

    @Test
    void init_featureEnabled_createsIndexAndScheduler() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        config.setChatIdAffinityExpirationMs(600_000);
        config.setChatIdAffinityMaxEntries(100);
        config.setChatIdAffinityCleanupIntervalMs(60_000);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        assertNotNull(service.getAffinityIndex());
        assertNotNull(service.getCleanupScheduler());
        assertFalse(service.getCleanupScheduler().isShutdown());

        service.shutdown();
        assertTrue(service.getCleanupScheduler().isShutdown());
    }

    @Test
    void recordAffinity_featureEnabled_delegatesToIndex() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        service.recordAffinity("chat-1");
        assertTrue(service.hasAffinity("chat-1"));
        assertEquals(1, service.getAffinityIndex().size());

        service.shutdown();
    }

    @Test
    void hasAffinity_featureEnabled_unknownChatId_returnsFalse() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        assertFalse(service.hasAffinity("unknown"));

        service.shutdown();
    }

    @Test
    void recordAffinity_nullChatId_isNoOp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        service.recordAffinity(null);
        assertEquals(0, service.getAffinityIndex().size());

        service.shutdown();
    }

    // --- Feature disabled ---

    @Test
    void init_featureDisabled_doesNotCreateIndexOrScheduler() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(false);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        assertNull(service.getAffinityIndex());
        assertNull(service.getCleanupScheduler());
    }

    @Test
    void recordAffinity_featureDisabled_isNoOp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(false);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        // Should not throw
        service.recordAffinity("chat-1");
    }

    @Test
    void hasAffinity_featureDisabled_returnsFalse() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(false);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        assertFalse(service.hasAffinity("chat-1"));
    }

    // --- Shutdown safety ---

    @Test
    void shutdown_featureDisabled_doesNotThrow() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(false);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        // Should not throw even when scheduler is null
        service.shutdown();
    }

    @Test
    void multipleRecordAndQuery_featureEnabled_worksCorrectly() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        ChatIdAffinityService service = new ChatIdAffinityService(configService);
        service.init();

        service.recordAffinity("chat-a");
        service.recordAffinity("chat-b");
        service.recordAffinity("chat-c");

        assertTrue(service.hasAffinity("chat-a"));
        assertTrue(service.hasAffinity("chat-b"));
        assertTrue(service.hasAffinity("chat-c"));
        assertFalse(service.hasAffinity("chat-d"));

        service.shutdown();
    }
}
