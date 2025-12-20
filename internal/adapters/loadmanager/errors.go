package loadmanager

import "errors"

// ErrQueueFull is returned when the request queue is at capacity
var ErrQueueFull = errors.New("server queue full, please retry after 30 seconds")
