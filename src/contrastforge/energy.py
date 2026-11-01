"""Generation-only GPU measurements. API-server energy is never inferred locally."""
import threading
import time


def integrate(samples):
    """Trapezoidal power integration: timestamps in seconds, power in watts."""
    if len(samples) < 2: return None
    if any(b[0] < a[0] for a,b in zip(samples,samples[1:])): raise ValueError("Nonmonotone timestamps")
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(samples,samples[1:]))


class EnergyMeter:
    def __init__(self, indices=None, interval=.1):
        self.indices, self.interval = indices, interval
        self.samples, self.result = [], {"joules":None,"method":"unavailable"}
        self.stop = threading.Event()

    def __enter__(self):
        if self.indices is None: return self
        try:
            import pynvml as nv
            nv.nvmlInit(); self.nv = nv
            self.handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in self.indices]
            self.result["device_uuids"] = [str(nv.nvmlDeviceGetUUID(h)) for h in self.handles]
            try: self.start_mj = [nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles]
            except nv.NVMLError: self.start_mj = None
            self._sample()
            self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
        except Exception as exc:
            self.result.update(method="unavailable", error=type(exc).__name__)
        return self

    def _sample(self):
        try:
            watts = sum(self.nv.nvmlDeviceGetPowerUsage(h)/1000 for h in self.handles)
            self.samples.append((time.perf_counter(),watts))
        except Exception:
            self.result["sample_error"] = True

    def _loop(self):
        while not self.stop.wait(self.interval): self._sample()

    def __exit__(self, *exc):
        if hasattr(self,"thread"):
            self.stop.set(); self.thread.join(timeout=2); self._sample()
            try:
                if self.start_mj is not None:
                    ends = [self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles]
                    diffs = [b-a for a,b in zip(self.start_mj,ends)]
                    if any(x<0 for x in diffs): raise ValueError("Energy counter reset")
                    self.result.update(joules=sum(diffs)/1000,method="nvml_total_energy_counter")
                elif not self.result.get("sample_error"):
                    self.result.update(joules=integrate(self.samples),method="nvml_power_trapezoid")
            except Exception as error:
                self.result.update(joules=None,method="unavailable",error=type(error).__name__)
            self.result["samples"] = len(self.samples)
            self.result["scope"] = "selected physical GPUs during synchronized generation; includes idle power"
            self.nv.nvmlShutdown()


def checked_nvml_indices(explicit, torch_device):
    """Require explicit physical indices and verify their UUID against the local CUDA device."""
    if explicit is None: return None
    import torch
    import pynvml as nv
    if len(explicit) != 1 or not str(torch_device).startswith("cuda"):
        raise ValueError("This collector measures one local CUDA GPU per process")
    nv.nvmlInit()
    try:
        uuid = str(torch.cuda.get_device_properties(torch_device).uuid)
        got = nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(explicit[0]))
        if isinstance(got,bytes): got=got.decode()
        if "MIG" in uuid or uuid.removeprefix("GPU-") != got.removeprefix("GPU-"):
            raise ValueError("NVML physical GPU UUID does not match the CUDA device; do not guess a device index")
    finally: nv.nvmlShutdown()
    return explicit
