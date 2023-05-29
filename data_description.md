
## Data description
From https://ronin.cs.sfu.ca/README.txt

Data sources
	 - gyro		 		- Android Sensor.TYPE_GYROSCOPE
	 - gyro_uncalib		- Android Sensor.TYPE_GYROSCOPE_UNCALIBRATED
	 - acce		 		- Android Sensor.TYPE_ACCELEROMETER
	 - linacce			- Android Sensor.TYPE_LINEAR_ACCELERATION
	 - gravity			- Android Sensor.TYPE_GRAVITY
	 - magnet			- Android Sensor.TYPE_MAGNETIC_FIELD
	 - magnet_uncalib 	- Android Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED
	 - rv		 		- Android Sensor.TYPE_ROTATION_VECTOR
	 - game_rv			- Android Sensor.TYPE_GAME_ROTATION_VECTOR
	 - magnetic_rv		- Android Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR
	 - step		 		- Android Sensor.TYPE_STEP_COUNTER
	 - pressure			- Android Sensor.TYPE_PRESSURE
	 - gps				- Android LocationManager.GPS_PROVIDER
	 - tango_pose		- Pose from Visual SLAM of Tango device (format: time, position (x,y,z), orientation (x,y,z,w))
	 - tango_adf__pose	- Pose from Visual SLAM with area learning of Tango device
	 - wifi				- wifi footprints scanned every 3 seconds. stored in 2 parts
		|-- "wifi_values" - contains (scan_number, last_timestep, level) 
		|--"wifi_address" - dataset of type string. contains BSSID of corresponding records in wifi_values

"raw" group contains data as reported by APIs in format 
	- system_timestamp (nanosecond), API output
"synced" group contains time synchronized data from IMU device sampled at 200 Hz
   	- time : System time of IMU device in seconds
"pose" group store all pose information (timestamp for data is "synced/time")
   	- tango_pos, tango_ori - tango_pose and tango_adf_pose combined for more accurate pose estimation
   	- ekf_ori - orientation of IMU device calculated using sensor fusion. (Should not be used during testing)