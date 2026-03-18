package org.flexlb.balance.resource;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.stereotype.Component;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Resource measure factory
 * Retrieves appropriate resource measure based on RoleType
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class ResourceMeasureFactory {

    private final Map<ResourceMeasureIndicatorEnum, ResourceMeasure> measureMap;

    public ResourceMeasureFactory(List<ResourceMeasure> measureList) {
        this.measureMap = new EnumMap<>(ResourceMeasureIndicatorEnum.class);
        for (ResourceMeasure measure : measureList) {
            measureMap.put(measure.getResourceMeasureIndicator(), measure);
        }
    }

    /**
     * Get resource measure based on resource indicator
     *
     * @param measureIndicator Resource measure indicator
     * @return Resource measure instance, or null if not found
     */
    public ResourceMeasure getMeasure(ResourceMeasureIndicatorEnum measureIndicator) {
        return measureMap.get(measureIndicator);
    }

    /**
     * Calculate the maximum water level across all role types.
     * Shared by RouteService (direct routing) and QueueManager (queue mode).
     *
     * @param config FlexlbConfig for role-to-indicator mapping
     * @return max water level (0-100) across all roles
     */
    public double calculateMaxWaterLevel(FlexlbConfig config) {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        List<RoleType> roleTypeList = modelWorkerStatus.getRoleTypeList();
        double maxWaterLevel = 0.0;
        for (RoleType roleType : roleTypeList) {
            Map<String, WorkerStatus> workerStatusMap = modelWorkerStatus.getRoleStatusMap(roleType);
            ResourceMeasureIndicatorEnum indicator = config.getResourceMeasureIndicator(roleType);
            ResourceMeasure measure = getMeasure(indicator);
            if (measure != null) {
                double waterLevel = measure.calculateAverageWaterLevel(workerStatusMap);
                maxWaterLevel = Math.max(maxWaterLevel, waterLevel);
            }
        }
        return maxWaterLevel;
    }
}
