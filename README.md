# inertial_navigation_transformer
Use transformer for inertial navigation

Typically IMU data is processed as shown in the flowchart below:


<figure>
  <img src="https://github.com/tonton-golio/inertial_navigation_transformer/blob/main/assets/IMU_interpretation_flowchart.png?raw=true" alt="IMU interpretation flowchart">
  <figcaption>IMU interpretation flowchart</figcaption>
</figure>



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



## links to get into the topic:
* The problem with IMU position estimation https://www.youtube.com/watch?v=_q_8d0E3tDk
* basic kalman filter introduction: https://www.youtube.com/watch?v=5HuN9iL-zxU
* get NED values for acc: https://www.youtube.com/watch?v=T9jXoG0QYIA
* strapdown: https://www.youtube.com/watch?v=knroQcjAgPo
* Romberg integration: https://www.youtube.com/watch?v=HtkylnbJklE
* paper on coning and sculling (integration techniques) https://sci-hub.hkvisa.net/10.2514/2.4718# This paper shows us how to convert a coning algorithm to its sculling algorithm equivalent, or vice versa. (probably no need to look at this)
* https://sci-hub.hkvisa.net/10.3390/s111009182 -- a great introduction!
* inertial navigation primer: https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-coning


## ideas
* throw raw acc, gyro and magnet at LSTM -> predict pose
  * hyperparameter-sweep
* transform raw inputs and measure the deacrease in training time, or increase in performance, or decrease in required NN size
* try transformer network
* Raw inegration for baseline
* kalman filter for better baseline (https://sci-hub.hkvisa.net/10.3390/s111009182)
* identify magnetic field disurbtions 
* single step (or few step) for neural network. 
  - Input vector is then: current state s_i (position and orientation) and imu readings d_{i:i+n}. 
  - Output is next state: s_n
* visualization tool (potentially 3d)
