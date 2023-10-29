import time
import os
import clr

from System import Action, Array
from System import *
from Mcs.Usb import CMcsUsbListNet, DeviceEnumNet, CStg200xDownloadNet, STG_DestinationEnumNet

clr.AddReference(os.getcwd() + 'library_files\\\McsUsbNet.dll')

class StimulationPattern:
    """
    Represents a stimulation pattern for a device.
    """

    def __poll_handler__(self, status, stg_status_net, index_list):
        print("%x %s" % (status, str(stg_status_net.TiggerStatus[0])))


    def __init__(self, durations, amplitudes,channel):
        """
        Initializes a new instance of the StimulationPattern class.

        Args:
            durations (list[float]): List of durations for specific channel in micro volt.
            amplitudes (list[float]): List of amplitudes for specific channel in micro second.
            channel (int): ID of the stimulator's output channel (0 or 1).
        """
        self.durations = durations
        self.amplitudes = amplitudes
        self.channel=channel

        self.device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        self.device = CStg200xDownloadNet()
        self.device.Stg200xPollStatusEvent += self.__poll_handler__
        self.device.Connect(self.device_list.GetUsbListEntry(0))
        self.device.SetVoltageMode()
        self.device.PrepareAndSendData(
            0, self.amplitudes, self.durations, STG_DestinationEnumNet.channeldata_voltage
        )
        self.device.SendStart(1)

    def download_stimulation(self):
        """
        Downloads the stimulation pattern to the device.
        """
        self.device.Connect(self.device_list.GetUsbListEntry(0))
        self.device.SetVoltageMode()
        self.device.PrepareAndSendData(
            self.channel, Array[Int32](self.amplitudes), Array[Int32](self.durations), STG_DestinationEnumNet.channeldata_voltage
        )
        self.device.SendStart(1)
        time.sleep(sum(self.durations)/10**6)
        self.device.Disconnect()


# TODO
# I have to check if it faces any problem for handler when we send stimulation over 2 channels.