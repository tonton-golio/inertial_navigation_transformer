# inertial_navigation_transformer
Use transformer for inertial navigation

Typically IMU data is processed as shown in the flowchart below:

![IMU interpretation flowchart](https://github.com/tonton-golio/inertial_navigation_transformer/blob/main/assets/IMU_interpretation_flowchart.png?raw=true)



The raw gyroscope and accelerometer readings enter on the left. The gyroscope data is then bias corrected and integrated to obtain a orientation change. 
> I have implemented the `utils.Theta(angular_velocity, timestepsize)` which returns a 4x4 matrix which updates the previous quaternion (see `playground_notebooks/quaternion_update_anton.ipynb` for example).

Note that at the first time step we need to determine the orientation in the world frame. The world frame is defined by the north, east and down directions.
> This is implemented in `playground_notebooks/bodyFrame_to_worldFrame.ipynb`.




## Todo
- [] list of material for research/introduction to dealing with IMU data.
- [] Implement neural network or for replacing `utils.Theta(angular_velocity, timestepsize)`. This should use `tango/ori` as ground truth, and take `synced/gyro` as inputs. For now, set the initial orientation equal to the groundtruth. Subsequently, implement larger network which also takes `synced/acce` and `synced/magnet` to predict world frame, instead of getting initial orientation from ground truth.
    * consider advantages of different architectures: feed-forward, RNN, LSTM, transformer.
    * We found that initial orientation introduces an issue
    * A dysfynctional version of this has been added to `playground_notebooks/quaternion_update_anton.ipynb`
- [] Make table of data contents, so we get an overview of what we are working with.
- [] set up github pages: https://docs.github.com/en/pages/quickstart

