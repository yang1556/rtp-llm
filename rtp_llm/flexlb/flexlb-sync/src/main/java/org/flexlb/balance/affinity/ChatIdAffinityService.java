package org.flexlb.balance.affinity;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.springframework.stereotype.Component;

/**
 * Unified interface for affinity management, wrapping ChatIdAffinityIndex operations.
 * When enableChatIdAffinity is false, all methods are no-ops.
 */
@Slf4j
@Component
public class ChatIdAffinityService {

    private final ConfigService configService;
    private volatile ChatIdAffinityIndex affinityIndex;
    private volatile ScheduledExecutorService cleanupScheduler;

    public ChatIdAffinityService(ConfigService configService) {
        this.configService = configService;
    }

    @PostConstruct
    void init() {
        FlexlbConfig config = configService.loadBalanceConfig();
        if (!config.isEnableChatIdAffinity()) {
            log.info("ChatId affinity is disabled, skipping initialization");
            return;
        }

        this.affinityIndex = new ChatIdAffinityIndex(
                config.getChatIdAffinityExpirationMs(),
                config.getChatIdAffinityMaxEntries()
        );

        this.cleanupScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "chatid-affinity-cleanup");
            t.setDaemon(true);
            return t;
        });

        long cleanupIntervalMs = config.getChatIdAffinityCleanupIntervalMs();
        cleanupScheduler.scheduleWithFixedDelay(
                affinityIndex::evictExpired,
                cleanupIntervalMs,
                cleanupIntervalMs,
                TimeUnit.MILLISECONDS
        );

        log.info("ChatId affinity initialized: expirationMs={}, maxEntries={}, cleanupIntervalMs={}",
                config.getChatIdAffinityExpirationMs(),
                config.getChatIdAffinityMaxEntries(),
                cleanupIntervalMs);
    }

    @PreDestroy
    void shutdown() {
        if (cleanupScheduler != null) {
            cleanupScheduler.shutdown();
            log.info("ChatId affinity cleanup scheduler shut down");
        }
    }

    /**
     * Records chatId affinity. No-op when feature is disabled.
     */
    public void recordAffinity(String chatId) {
        if (affinityIndex == null) {
            return;
        }
        affinityIndex.put(chatId);
    }

    /**
     * Queries whether a chatId has affinity. Returns false when feature is disabled.
     */
    public boolean hasAffinity(String chatId) {
        if (affinityIndex == null) {
            return false;
        }
        return affinityIndex.hasAffinity(chatId);
    }

    // Visible for testing
    ChatIdAffinityIndex getAffinityIndex() {
        return affinityIndex;
    }

    // Visible for testing
    ScheduledExecutorService getCleanupScheduler() {
        return cleanupScheduler;
    }
}
