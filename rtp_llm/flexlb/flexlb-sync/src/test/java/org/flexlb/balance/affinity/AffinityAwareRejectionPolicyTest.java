package org.flexlb.balance.affinity;

import org.flexlb.balance.affinity.AffinityAwareRejectionPolicy.RejectionReason;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AffinityAwareRejectionPolicyTest {

    @Mock
    private ChatIdAffinityService affinityService;

    @Mock
    private ConfigService configService;

    private AffinityAwareRejectionPolicy policy;

    private FlexlbConfig enabledConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(true);
        return config;
    }

    private FlexlbConfig disabledConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEnableChatIdAffinity(false);
        return config;
    }

    private Request requestWithChatId(String chatId) {
        Request request = new Request();
        request.setChatId(chatId);
        return request;
    }

    @BeforeEach
    void setUp() {
        policy = new AffinityAwareRejectionPolicy(affinityService, configService);
    }

    // ========== shouldRejectRequest tests ==========

    @Nested
    class ShouldRejectRequestTests {

        @Test
        void featureDisabled_returnsTrue() {
            when(configService.loadBalanceConfig()).thenReturn(disabledConfig());

            assertTrue(policy.shouldRejectRequest(requestWithChatId("chat-1"), RejectionReason.QUEUE_FULL));
            assertTrue(policy.shouldRejectRequest(requestWithChatId("chat-1"), RejectionReason.NO_AVAILABLE_WORKER));
        }

        @Test
        void nullChatId_returnsTrue() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertTrue(policy.shouldRejectRequest(requestWithChatId(null), RejectionReason.QUEUE_FULL));
        }

        @Test
        void emptyChatId_returnsTrue() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertTrue(policy.shouldRejectRequest(requestWithChatId(""), RejectionReason.QUEUE_FULL));
        }

        @Test
        void noAffinity_returnsTrue() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            assertTrue(policy.shouldRejectRequest(requestWithChatId("chat-new"), RejectionReason.QUEUE_FULL));
        }

        @Test
        void hasAffinity_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-existing")).thenReturn(true);

            assertFalse(policy.shouldRejectRequest(requestWithChatId("chat-existing"), RejectionReason.QUEUE_FULL));
        }

        @Test
        void hasAffinity_noAvailableWorker_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-existing")).thenReturn(true);

            assertFalse(policy.shouldRejectRequest(requestWithChatId("chat-existing"), RejectionReason.NO_AVAILABLE_WORKER));
        }

        @Test
        void noAffinity_noAvailableWorker_returnsTrue() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            assertTrue(policy.shouldRejectRequest(requestWithChatId("chat-new"), RejectionReason.NO_AVAILABLE_WORKER));
        }
    }

    // ========== shouldRejectByWaterLevel tests ==========

    @Nested
    class ShouldRejectByWaterLevelTests {

        private static final double THRESHOLD = 70.0;

        @Test
        void featureDisabled_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(disabledConfig());

            // Should return false regardless of water level
            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 50.0, THRESHOLD));
            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 80.0, THRESHOLD));
            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 100.0, THRESHOLD));
        }

        // --- Water level below threshold ---

        @Test
        void waterLevelBelowThreshold_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 50.0, THRESHOLD));
        }

        @Test
        void waterLevelZero_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 0.0, THRESHOLD));
        }

        @Test
        void waterLevelJustBelowThreshold_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 69.9, THRESHOLD));
        }

        // --- Water level at threshold: rejectionRate = 0%, should never reject ---

        @Test
        void waterLevelAtThreshold_noAffinity_neverRejects() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            // rejectionRate = (70 - 70) / (100 - 70) = 0.0 → never rejects
            for (int i = 0; i < 100; i++) {
                assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 70.0, THRESHOLD));
            }
        }

        @Test
        void waterLevelAtThreshold_hasAffinity_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-existing")).thenReturn(true);

            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-existing"), 70.0, THRESHOLD));
        }

        // --- Water level in soft zone: probabilistic rejection for non-affinity ---

        @Test
        void waterLevelInSoftZone_noAffinity_probabilisticRejection() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            // waterLevel=85, threshold=70 → rejectionRate = (85-70)/(100-70) = 50%
            int trials = 1000;
            int rejections = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 85.0, THRESHOLD)) {
                    rejections++;
                }
            }
            // Expect ~50% rejection rate, allow wide margin for randomness
            double rate = (double) rejections / trials;
            assertTrue(rate > 0.3 && rate < 0.7,
                    "Expected ~50% rejection rate at waterLevel=85, got " + rate);
        }

        @Test
        void waterLevelInSoftZone_hasAffinity_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-existing")).thenReturn(true);

            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-existing"), 85.0, THRESHOLD));
        }

        @Test
        void waterLevelAt99_noAffinity_almostAlwaysRejects() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            // waterLevel=99.9, threshold=70 → rejectionRate = (99.9-70)/(100-70) ≈ 99.7%
            int trials = 1000;
            int rejections = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 99.9, THRESHOLD)) {
                    rejections++;
                }
            }
            double rate = (double) rejections / trials;
            assertTrue(rate > 0.95,
                    "Expected >95% rejection rate at waterLevel=99.9, got " + rate);
        }

        @Test
        void waterLevelAt99_hasAffinity_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-existing")).thenReturn(true);

            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-existing"), 99.9, THRESHOLD));
        }

        // --- Smooth rejection rate increases linearly with water level ---

        @Test
        void rejectionRate_increasesWithWaterLevel() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            int trials = 2000;

            // waterLevel=75 → rejectionRate = (75-70)/(100-70) ≈ 16.7%
            int rejectionsLow = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 75.0, THRESHOLD)) {
                    rejectionsLow++;
                }
            }

            // waterLevel=95 → rejectionRate = (95-70)/(100-70) ≈ 83.3%
            int rejectionsHigh = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 95.0, THRESHOLD)) {
                    rejectionsHigh++;
                }
            }

            assertTrue(rejectionsHigh > rejectionsLow,
                    "Higher water level should produce more rejections: low=" + rejectionsLow + " high=" + rejectionsHigh);
        }

        // --- Water level at or above 100 (reject all) ---

        @Test
        void waterLevel100_rejectsAll() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertTrue(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 100.0, THRESHOLD));
        }

        @Test
        void waterLevelAbove100_rejectsAll() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            assertTrue(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 105.0, THRESHOLD));
        }

        // --- Null/empty chatId in soft rejection zone: probabilistic rejection ---

        @Test
        void softZone_nullChatId_probabilisticRejection() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            // null chatId has no affinity → subject to probabilistic rejection
            // waterLevel=85, threshold=70 → rejectionRate = 50%
            int trials = 200;
            int rejections = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId(null), 85.0, THRESHOLD)) {
                    rejections++;
                }
            }
            double rate = (double) rejections / trials;
            assertTrue(rate > 0.2 && rate < 0.8,
                    "Expected ~50% rejection rate for null chatId at waterLevel=85, got " + rate);
        }

        @Test
        void softZone_emptyChatId_probabilisticRejection() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            int trials = 200;
            int rejections = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId(""), 85.0, THRESHOLD)) {
                    rejections++;
                }
            }
            double rate = (double) rejections / trials;
            assertTrue(rate > 0.2 && rate < 0.8,
                    "Expected ~50% rejection rate for empty chatId at waterLevel=85, got " + rate);
        }

        // --- Threshold boundary cases ---

        @Test
        void thresholdZero_anyPositiveWaterLevel_probabilisticRejection() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            // threshold=0, waterLevel=50 → rejectionRate = (50-0)/(100-0) = 50%
            int trials = 500;
            int rejections = 0;
            for (int i = 0; i < trials; i++) {
                if (policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 50.0, 0.0)) {
                    rejections++;
                }
            }
            double rate = (double) rejections / trials;
            assertTrue(rate > 0.3 && rate < 0.7,
                    "Expected ~50% rejection rate at waterLevel=50 threshold=0, got " + rate);
        }

        @Test
        void thresholdZero_waterLevelZero_neverRejects() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());
            when(affinityService.hasAffinity("chat-new")).thenReturn(false);

            // threshold=0, waterLevel=0 → rejectionRate = (0-0)/(100-0) = 0%
            for (int i = 0; i < 100; i++) {
                assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-new"), 0.0, 0.0));
            }
        }

        @Test
        void threshold100_waterLevelBelow100_returnsFalse() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            // threshold=100, waterLevel=99 → waterLevel < threshold → false
            assertFalse(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 99.0, 100.0));
        }

        @Test
        void threshold100_waterLevel100_rejectsAll() {
            when(configService.loadBalanceConfig()).thenReturn(enabledConfig());

            // threshold=100, waterLevel=100 → waterLevel >= 100 → true
            assertTrue(policy.shouldRejectByWaterLevel(requestWithChatId("chat-1"), 100.0, 100.0));
        }
    }
}
