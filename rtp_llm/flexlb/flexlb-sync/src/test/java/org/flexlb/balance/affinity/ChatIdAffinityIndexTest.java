package org.flexlb.balance.affinity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatIdAffinityIndexTest {

    private ChatIdAffinityIndex index;

    @BeforeEach
    void setUp() {
        // 10 min expiration, max 5 entries for easier capacity testing
        index = new ChatIdAffinityIndex(600_000L, 5);
    }

    // --- put / hasAffinity basic ---

    @Test
    void putAndHasAffinity_validChatId_returnsTrue() {
        index.put("chat-1");
        assertTrue(index.hasAffinity("chat-1"));
    }

    @Test
    void hasAffinity_unknownChatId_returnsFalse() {
        assertFalse(index.hasAffinity("unknown"));
    }

    @Test
    void put_updateExistingChatId_refreshesTimestamp() {
        index.put("chat-1");
        assertTrue(index.hasAffinity("chat-1"));
        // put again should still work
        index.put("chat-1");
        assertTrue(index.hasAffinity("chat-1"));
        assertEquals(1, index.size());
    }

    // --- null / empty input ---

    @Test
    void put_nullChatId_isNoOp() {
        index.put(null);
        assertEquals(0, index.size());
    }

    @Test
    void put_emptyChatId_isNoOp() {
        index.put("");
        assertEquals(0, index.size());
    }

    @Test
    void hasAffinity_nullChatId_returnsFalse() {
        assertFalse(index.hasAffinity(null));
    }

    @Test
    void hasAffinity_emptyChatId_returnsFalse() {
        assertFalse(index.hasAffinity(""));
    }

    // --- expiration ---

    @Test
    void hasAffinity_expiredEntry_returnsFalse() {
        // Use a very short expiration
        ChatIdAffinityIndex shortLived = new ChatIdAffinityIndex(1L, 1000);
        shortLived.put("chat-1");
        // Sleep just enough for expiration
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        assertFalse(shortLived.hasAffinity("chat-1"));
    }

    // --- evictExpired ---

    @Test
    void evictExpired_removesExpiredEntries() {
        ChatIdAffinityIndex shortLived = new ChatIdAffinityIndex(1L, 1000);
        shortLived.put("chat-1");
        shortLived.put("chat-2");
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        shortLived.evictExpired();
        assertEquals(0, shortLived.size());
    }

    @Test
    void evictExpired_keepsNonExpiredEntries() {
        index.put("chat-1");
        index.put("chat-2");
        index.evictExpired();
        assertEquals(2, index.size());
        assertTrue(index.hasAffinity("chat-1"));
        assertTrue(index.hasAffinity("chat-2"));
    }

    // --- capacity eviction ---

    @Test
    void put_exceedingMaxEntries_evictsOldestEntry() {
        // maxEntries = 5, use explicit timestamps to ensure deterministic ordering
        long baseTime = System.currentTimeMillis();
        for (int i = 1; i <= 5; i++) {
            index.put("chat-" + i, baseTime + i);
        }
        assertEquals(5, index.size());

        // Adding 6th should evict the oldest (chat-1, timestamp = baseTime+1)
        index.put("chat-6", baseTime + 6);
        assertTrue(index.size() <= 5 + 1); // size() ≤ maxEntries + 1 in steady state
        // chat-1 was the oldest, should be evicted
        assertFalse(index.hasAffinity("chat-1"));
        assertTrue(index.hasAffinity("chat-6"));
    }

    @Test
    void put_multipleOverCapacity_maintainsBound() {
        // maxEntries = 5, add 10 entries with distinct timestamps
        long baseTime = System.currentTimeMillis();
        for (int i = 1; i <= 10; i++) {
            index.put("chat-" + i, baseTime + i);
        }
        // After all puts, size should be at most maxEntries + 1
        assertTrue(index.size() <= 6, "size should be bounded: " + index.size());
    }

    // --- size ---

    @Test
    void size_emptyIndex_returnsZero() {
        assertEquals(0, index.size());
    }

    @Test
    void size_afterPuts_returnsCorrectCount() {
        index.put("a");
        index.put("b");
        index.put("c");
        assertEquals(3, index.size());
    }

    // --- concurrent access ---

    @Test
    void concurrentPutAndHasAffinity_noExceptions() throws InterruptedException {
        ChatIdAffinityIndex concurrentIndex = new ChatIdAffinityIndex(600_000L, 1000);
        int threadCount = 10;
        int opsPerThread = 100;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        List<Throwable> errors = new ArrayList<>();

        for (int t = 0; t < threadCount; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    for (int i = 0; i < opsPerThread; i++) {
                        String chatId = "chat-" + threadId + "-" + i;
                        concurrentIndex.put(chatId);
                        concurrentIndex.hasAffinity(chatId);
                        // Also do some eviction
                        if (i % 20 == 0) {
                            concurrentIndex.evictExpired();
                        }
                    }
                } catch (Throwable e) {
                    synchronized (errors) {
                        errors.add(e);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        assertTrue(latch.await(10, TimeUnit.SECONDS), "Threads should complete within timeout");
        executor.shutdown();
        assertTrue(errors.isEmpty(), "No exceptions expected, got: " + errors);
    }

    @Test
    void concurrentPutOnSameChatId_noDataLoss() throws InterruptedException {
        ChatIdAffinityIndex concurrentIndex = new ChatIdAffinityIndex(600_000L, 1000);
        int threadCount = 10;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);

        for (int t = 0; t < threadCount; t++) {
            executor.submit(() -> {
                try {
                    concurrentIndex.put("shared-chat");
                } finally {
                    latch.countDown();
                }
            });
        }

        assertTrue(latch.await(5, TimeUnit.SECONDS));
        executor.shutdown();
        assertTrue(concurrentIndex.hasAffinity("shared-chat"));
        assertEquals(1, concurrentIndex.size());
    }
}
