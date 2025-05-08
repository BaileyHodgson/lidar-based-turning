#include "a1lidarrpi.h"
#include <vector>
#include <cmath>
#include <cstdio>

// ---------- CONFIGURATION CONSTANTS ----------

// Edge detection sensitivity (difference in range between two points to be considered an edge)
constexpr float edgeThreshold = 0.25f;  // meters

// Define the angular window (in degrees) in front of the robot to look for edges
constexpr float frontAngleMin = -15.0f; // degrees (left edge of forward cone)
constexpr float frontAngleMax = 15.0f;  // degrees (right edge of forward cone)

// Global flag to avoid multiple turns per session
bool edgeDetected = false;

// ---------- UTILITY FUNCTION: Check if angle is in front sector ----------
bool inFront(float angleRadians) {
	float deg = angleRadians * 180.0f / M_PI; // Convert radians to degrees
	return deg >= frontAngleMin && deg <= frontAngleMax;
}

// ---------- FUNCTION: Simulate a 90° robot turn (GPIO control placeholder) ----------
void rotate90Degrees() {
	fprintf(stderr, ">>> Rotating robot 90 degrees...\n");
}

// ---------- CUSTOM LIDAR DATA HANDLER ----------
class DataInterface : public A1Lidar::DataInterface {
    public:
        bool turning = false;
        float referenceAngle = 0.0f; // Angle to compare against
        bool referenceCaptured = false;
    
        // Called for every new 360° scan
        void newScanAvail(float, A1LidarData (&data)[A1Lidar::nDistance]) override {
            // Step 1: Get angle of closest valid object
            float minRange = 1000.0f;  // Some large initial number
            float angleAtMinRange = 0.0f;
    
            for (A1LidarData &point : data) {
                if (point.valid && point.r < minRange) {
                    minRange = point.r;
                    angleAtMinRange = point.phi;
                }
            }
    
            // Step 2: Capture reference before turning
            if (!referenceCaptured && minRange < 5.0f) {
                referenceAngle = angleAtMinRange;
                referenceCaptured = true;
                fprintf(stderr, "\nReference angle captured: %.2f°\n", angleAtMinRange * 180.0f / M_PI);
    
                // Start turning
                turning = true;
                fprintf(stderr, ">>> START TURN\n");
            }
    
            // Step 3: While turning, check angular difference
            if (turning) {
                float angleDiff = angleAtMinRange - referenceAngle;
    
                // Normalize to [-PI, PI]
                while (angleDiff > M_PI) angleDiff -= 2*M_PI;
                while (angleDiff < -M_PI) angleDiff += 2*M_PI;
    
                float degDiff = angleDiff * 180.0f / M_PI;
                fprintf(stderr, "Angle diff: %.2f°\n", degDiff);
    
                if (std::abs(degDiff) >= 90.0f) {
                    // Reached 90° turn
                    turning = false;
    
                    // Stop motors
                    // digitalWrite(LEFT_MOTOR_PIN, LOW);
                    // digitalWrite(RIGHT_MOTOR_PIN, LOW);
                    fprintf(stderr, ">>> STOP TURN at %.2f° diff\n", degDiff);
                }
            }
        }
    };
    



// ---------- MAIN FUNCTION ----------
int main(int, char **) {
	// Print info about data format
	fprintf(stderr,"Data format: x <tab> y <tab> r <tab> phi <tab> strength\n");
	fprintf(stderr,"Press any key to stop.\n");

	A1Lidar lidar;                // Create a LIDAR object
	DataInterface dataInterface; // Create the custom data handler

	lidar.registerInterface(&dataInterface); // Register the callback
	lidar.start();                            // Begin scanning

	// Wait until user presses any key
	do {
	} while (!getchar());

	// Stop LIDAR and clean up
	lidar.stop();
	fprintf(stderr,"\n");
	return 0;
}
