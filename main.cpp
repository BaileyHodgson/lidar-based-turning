#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <a1lidarrpi.h>  // Include the LIDAR header file

using namespace std;

struct ScanData {
    std::vector<float> distances;
};

// Normalize angle to range [0, 359]
int normalize_angle(float angle) {
    int a = static_cast<int>(angle);
    a = ((a % 360) + 360) % 360;
    return a;
}

// Compute edge energy (sum of absolute differences between distances)
float compute_edge_energy(const std::vector<float>& distances) {
    float energy = 0.0f;
    for (size_t i = 1; i < distances.size(); ++i) {
        if (distances[i] > 0 && distances[i - 1] > 0) {
            energy += std::abs(distances[i] - distances[i - 1]);
        }
    }
    return energy;
}

// Compute the total difference between two scans (point-wise absolute difference)
float compute_difference(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > 0 && b[i] > 0) {
            diff += std::abs(a[i] - b[i]);
        }
    }
    return diff;
}

class RotationDetector {
public:
    RotationDetector(A1Lidar& lidar) : lidar_(lidar) {}

    // Get data from LIDAR and convert it into a ScanData format
    bool acquire_scan(ScanData& scan) {
        // LIDAR data is stored in `a1LidarData`
        lidar_.getData();  // Get new data from LIDAR

        if (lidar_.dataAvailable) {
            // Extract distances from the LIDAR data and fill the ScanData structure
            scan.distances.clear();
            scan.distances.resize(360, 0.0f);  // Initialize distances for 360 degrees

            for (int i = 0; i < 360; ++i) {
                if (lidar_.a1LidarData[lidar_.currentBufIdx][i].valid) {
                    scan.distances[i] = lidar_.a1LidarData[lidar_.currentBufIdx][i].r;
                }
            }
            return true;
        }
        return false;
    }

    // Detect rotation by comparing current scan with baseline scan
    bool detect_rotation(ScanData& baseline_scan) {
        ScanData current_scan;
        if (acquire_scan(current_scan)) {
            float diff = compute_difference(baseline_scan.distances, current_scan.distances);
            std::cout << "Difference from baseline: " << diff << std::endl;

            // Threshold to detect significant rotation (adjust as needed)
            if (diff > 8000.0f) {
                std::cout << "Rotation detected!" << std::endl;
                return true;
            }
        }
        return false;
    }

private:
    A1Lidar& lidar_;
};

int main() {
    const char* serial_port = "/dev/serial0";  // Use GPIO UART for communication
    unsigned int rpm = 600;  // Set desired RPM

    // Initialize the LIDAR object
    A1Lidar lidar;
    lidar.start(serial_port, rpm);  // Start the LIDAR scan

    // Create a rotation detector object
    RotationDetector rotation_detector(lidar);

    // Capture the baseline scan
    ScanData baseline_scan;
    while (!rotation_detector.acquire_scan(baseline_scan)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Monitoring for rotation..." << std::endl;

    // Continuously monitor for significant rotation based on edge detection
    while (true) {
        if (rotation_detector.detect_rotation(baseline_scan)) {
            // Handle rotation detected
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));  // Delay between scans
    }

    // Stop the LIDAR when done
    lidar.stop();
    return 0;
}
