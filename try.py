from fare_net import FARE_Net

enhanced_path = "log/enhanced/60-1"
enhanced_fare_net = FARE_Net(far_net_model_path=f"{enhanced_path}/far_net_model.keras", e_net_model_path=f"{enhanced_path}/e_net_model.keras")

enhanced_fare_net.visualize_gaze(camera_index=1)