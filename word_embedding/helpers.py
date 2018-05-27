class TimeDiff:
    """Helper class to handle a duration in various units.
    
    Attributes:
        time_diff (int): duration to manage (in seconds).
    """
    def __init__(self, time1, time2):
        """Initialize the duration.
        
        Args:
            time1 (int): first timestamp in second.
            time2 (int): second timestamp in second.
        """
        self.time_diff = time2-time1

    def hms(self):
        """Return the number of hours, minutes and seconds from the duration."""
        hours = self.time_diff // 3600
        minutes = (self.time_diff % 3600) // 60
        seconds = self.time_diff % 60
        return [int(hours), int(minutes), int(seconds)]

    def ms(self):
        """Return the number of minutes and seconds from the duration."""
        minutes = self.time_diff // 60
        seconds = self.time_diff % 60
        return [int(minutes), int(seconds)]
    
    def s(self):
        """Return the number of seconds from the duration."""
        return int(self.time_diff)