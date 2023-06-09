import dask.array as da

def events_in_circle(events, radius, center):
    """Filter events in a circle.
    """

    x = events[:, 1]
    y = events[:, 2]
    idx = (x - center[0])**2 + (y - center[1])**2 < radius**2
    roi_size = (2 * radius, 2 * radius)
    return events[idx].compute_chunk_sizes(), roi_size

def downsample_events(events, downsampling_factor):
    """Downsample events by a factor.
    """
    if downsampling_factor == 1:
        return events
    else:
        return events[::downsampling_factor].compute_chunk_sizes()
    
def iter_minutes(events):
    """Iterate over events in minutes.
    """
    timestamps = events[:, 2]
    # generate timestamps of all minutes using linspace
    minutes = da.linspace(timestamps[0], timestamps[-1], (timestamps[-1] - timestamps[0]) // 60e9 + 1)
    # find indices of events in each minute
    idx = da.searchsorted(timestamps, minutes)
    # iterate over minutes
    for i in range(len(minutes) - 1):
        yield events[idx[i]:idx[i+1]]

def iter_periods(events, period):
    """Iterate over events in periods.
    """
    timestamps = events[:, 2]
    # generate timestamps of all periods using linspace
    periods = da.linspace(timestamps[0], timestamps[-1], da.floor((timestamps[-1] - timestamps[0]) / (period*1e9)) + 1)
    # find indices of events in each period
    idx = da.searchsorted(timestamps, periods).compute()
    # iterate over periods
    for i in range(len(periods) - 1):
        yield events[idx[i]:idx[i+1]]

    

