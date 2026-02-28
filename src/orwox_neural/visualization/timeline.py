class TimelineRecorder:
    def __init__(self):
        self.events = []

    def record(self, layer_name, event_type, data):
        self.events.append({
            "layer": layer_name,
            "event": event_type,
            "data": data
        })

    def get_timeline(self):
        return self.events
