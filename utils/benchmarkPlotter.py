from results import raw_results
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

class device:
    def __init__(self, raw_res):
        self.name = raw_res["name"]   
        self.results = raw_res["results"]

    def __getitem__(self, name):
        return self.results[name]

    def __str__(self):
        return self.name

    def getFPS(self, name):
        return self.results[name][0]

    def getLatency(self, name):
        return self.results[name][1]

def main():
    results = []
    for raw_res in raw_results:
        results.append(device(raw_res))
    model_names = ["b12", "b12-quan", "b10", "b10-quan", "scrfd", "scrfd-quan"]

    print("FPS vs Devices\n========================")
    figure(figsize=(12, 10), dpi=120)
    for model_name in model_names:
        print(f"Plotting model {model_name}")
        device_names = [str(results[i]) for i in range(len(results))]
        fps = [results[i].getFPS(model_name) for i in range(len(results))]
        plt.plot(device_names, fps, label=model_name)

    plt.xlabel("Device")
    plt.ylabel("FPS")
    plt.title("FPS comparison on different devices")
    plt.grid()
    plt.legend()
    print("Saving...")
    plt.savefig("fps-vs-device.png")
    print("========================")
    #######################################################################################
    print("Latency vs Devices\n========================")
    figure(figsize=(12, 10), dpi=120)
    for model_name in model_names:
        print(f"Plotting model {model_name}")
        latency = [results[i].getLatency(model_name) for i in range(len(results))]
        plt.plot(device_names, latency, label=model_name)

    plt.xlabel("Device")
    plt.ylabel("latency (ms)")
    plt.title("Latency comparison on different devices")
    plt.grid()
    plt.legend()
    print("Saving...")
    plt.savefig("latency-vs-device.png")
    print("========================")
    #######################################################################################
    print("FPS speedup vs Devices\n========================")
    model_names = ["b12", "b10", "scrfd"]

    figure(figsize=(12, 10), dpi=120)
    for i, device_name in enumerate(device_names):
        print(f"Plotting FPS speed up for device {device_name}")
        fps_speedup = [ (results[i].getFPS(model_name+"-quan") - results[i].getFPS(model_name)) / results[i].getFPS(model_name) * 100 
                        for model_name in model_names]
        print(fps_speedup)
        plt.plot(model_names, fps_speedup, label=device_name)
    plt.xlabel("Model")
    plt.ylabel("speedup (%)")
    plt.title("FPS speedup comparison on different devices")
    plt.grid()
    plt.legend()
    print("Saving...")
    plt.savefig("fps-speedup-vs-model.png")
    print("========================")
    #######################################################################################
    print("Latency speedup vs Devices\n========================")
    figure(figsize=(12, 10), dpi=120)
    for i, device_name in enumerate(device_names):
        print(f"Plotting Latency speed up for device {device_name}")
        latency_speedup = [ -(results[i].getLatency(model_name+"-quan") - results[i].getLatency(model_name)) / results[i].getLatency(model_name) * 100 
                        for model_name in model_names]
        print(latency_speedup)
        plt.plot(model_names, latency_speedup, label=device_name)
    plt.xlabel("Model")
    plt.ylabel("speedup (%)")
    plt.title("Latency speedup comparison on different devices")
    plt.grid()
    plt.legend()
    print("Saving...")
    plt.savefig("latency-speedup-vs-model.png")
    print("========================")

if __name__ == '__main__':
    main()
    print("Done!")
