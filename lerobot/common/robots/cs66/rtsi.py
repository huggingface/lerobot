import struct
import socket
import select
import sys
import logging

from . import serialize

DEFAULT_TIMEOUT = 10.0

LOGNAME = 'rtsi'
_log = logging.getLogger(LOGNAME)


class Command:
    RTSI_REQUEST_PROTOCOL_VERSION = 86        # ascii V
    RTSI_GET_ELITECONTROL_VERSION = 118          # ascii v
    RTSI_TEXT_MESSAGE = 77                    # ascii M
    RTSI_DATA_PACKAGE = 85                    # ascii U
    RTSI_CONTROL_PACKAGE_SETUP_OUTPUTS = 79   # ascii O
    RTSI_CONTROL_PACKAGE_SETUP_INPUTS = 73    # ascii I
    RTSI_CONTROL_PACKAGE_START = 83           # ascii S
    RTSI_CONTROL_PACKAGE_PAUSE = 80           # ascii P

RTSI_PROTOCOL_VERSION_1 = 1

class ConnectionState:
    DISCONNECTED = 0
    CONNECTED = 1
    STARTED = 2
    PAUSED = 3

class RTSIException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

class RTSITimeoutException(RTSIException):
    def __init__(self, msg):
        super(RTSITimeoutException, self).__init__(msg)

class rtsi(object):
    def __init__(self, hostname, port=30004):
        self.hostname = hostname
        self.port = port
        self.__conn_state = ConnectionState.DISCONNECTED
        self.__sock = None
        self.__output_config = {}
        self.__input_config = {}
        self.__skipped_package_count = 0
        self.__protocolVersion = RTSI_PROTOCOL_VERSION_1

    def connect(self):
        if self.__sock:
            return

        self.__buf = b'' # buffer data in binary format
        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.__sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.__sock.settimeout(DEFAULT_TIMEOUT)
            self.__skipped_package_count = 0
            self.__sock.connect((self.hostname, self.port))
            self.__conn_state = ConnectionState.CONNECTED
        except (socket.timeout, socket.error):
            self.__sock = None
            raise

    def disconnect(self):
        if self.__sock:
            self.__sock.close()
            self.__sock = None
        self.__conn_state = ConnectionState.DISCONNECTED

    def is_connected(self):
        return self.__conn_state is not ConnectionState.DISCONNECTED

    def controller_version(self):
        cmd = Command.RTSI_GET_ELITECONTROL_VERSION
        version = self.__sendAndReceive(cmd)
        if version:
            _log.info('Controller version: ' + str(version.major) + '.' + str(version.minor) + '.' + str(version.bugfix)+ '.' + str(version.build))
            if version.major == 3 and version.minor <= 2 and version.bugfix < 19171:
                _log.error("Please upgrade your controller to minimally version 2.10")
                sys.exit()
            return version.major, version.minor, version.bugfix, version.build
        return None, None, None, None

    def version_check(self):
        cmd = Command.RTSI_REQUEST_PROTOCOL_VERSION
        payload = struct.pack('>H', RTSI_PROTOCOL_VERSION_1)
        success = self.__sendAndReceive(cmd, payload)
        if success:
            self.__protocolVersion = RTSI_PROTOCOL_VERSION_1
        return success

    def input_subscribe(self, variables:str):
        cmd = Command.RTSI_CONTROL_PACKAGE_SETUP_INPUTS
        payload = variables.encode()
        result = self.__sendAndReceive(cmd, payload)
        result.names = variables.split(',')
        self.__input_config[result.id] = result
        return serialize.DataObject.create_empty(variables, result.id)

    def output_subscribe(self, variables:str, frequency=125):
        cmd = Command.RTSI_CONTROL_PACKAGE_SETUP_OUTPUTS
        payload = struct.pack('>d', frequency)
        payload = payload + variables.encode()
        result = self.__sendAndReceive(cmd, payload)
        result.names = variables.split(',')
        self.__output_config[result.id] = result
        return result

    def start(self):
        cmd = Command.RTSI_CONTROL_PACKAGE_START
        success = self.__sendAndReceive(cmd)
        if success:
            _log.info('RTSI synchronization started')
            self.__conn_state = ConnectionState.STARTED
        else:
            _log.error('RTSI synchronization failed to start')
        return success

    def pause(self):
        cmd = Command.RTSI_CONTROL_PACKAGE_PAUSE
        success = self.__sendAndReceive(cmd)
        if success:
            _log.info('RTSI synchronization paused')
            self.__conn_state = ConnectionState.PAUSED
        else:
            _log.error('RTSI synchronization failed to pause')
        return success

    def set_input(self, input_data):
        if self.__conn_state != ConnectionState.STARTED:
            _log.error('Cannot send when RTSI synchronization is inactive')
            return
        if not input_data.recipe_id in self.__input_config:
            _log.error('Input configuration id not found: ' + str(input_data.recipe_id))
            return
        config = self.__input_config[input_data.recipe_id]
        return self.__sendall(Command.RTSI_DATA_PACKAGE, config.pack(input_data))


    def get_output_data(self):
        """Recieve the latest data package.
        If muliple packages has been received, older ones are discarded
        and only the newest one will be returned. Will block untill a package
        is received or the connection is lost
        """
        if self.__conn_state != ConnectionState.STARTED:
            raise RTSIException('Cannot receive when RTSI synchronization is inactive')
        return self.__recv(Command.RTSI_DATA_PACKAGE, False)

    def get_output_data_buffered(self, buffer_limit = None):
        """Recieve the next data package.
        If muliple packages has been received they are buffered and will
        be returned on subsequent calls to this function.
        Returns None if no data is available.
        """

        if self._rtsi__output_config is None:
            logging.error("Output configuration not initialized")
            return None

        try:
            while (
                self.is_connected()
                and (buffer_limit == None or len(self.__buf) < buffer_limit)
                and self.__recv_to_buffer(0)
            ):
                pass
        except RTSIException as e:
            data = self.__recv_from_buffer(Command.RTSI_DATA_PACKAGE, False)
            if data == None:
                raise e
        else:
            data = self.__recv_from_buffer(Command.RTSI_DATA_PACKAGE, False)

        return data

    def send_message(self, message, source = b"Python Client", type = serialize.Message.INFO_MESSAGE):
        cmd = Command.RTSI_TEXT_MESSAGE
        fmt = '>B%dsB%dsB' % (len(message), len(source))
        payload = struct.pack(fmt, len(message), message, len(source), source, type)
        return self.__sendall(cmd, payload)

    def __on_packet(self, cmd, payload):
        if cmd == Command.RTSI_REQUEST_PROTOCOL_VERSION:
            return self.__unpack_protocol_version_package(payload)
        elif cmd == Command.RTSI_GET_ELITECONTROL_VERSION:
            return self.__unpack_elitecontrol_version_package(payload)
        elif cmd == Command.RTSI_TEXT_MESSAGE:
            return self.__unpack_text_message(payload)
        elif cmd == Command.RTSI_CONTROL_PACKAGE_SETUP_OUTPUTS:
            return self.__unpack_setup_outputs_package(payload)
        elif cmd == Command.RTSI_CONTROL_PACKAGE_SETUP_INPUTS:
            return self.__unpack_setup_inputs_package(payload)
        elif cmd == Command.RTSI_CONTROL_PACKAGE_START:
            return self.__unpack_start_package(payload)
        elif cmd == Command.RTSI_CONTROL_PACKAGE_PAUSE:
            return self.__unpack_pause_package(payload)
        elif cmd == Command.RTSI_DATA_PACKAGE:
            return self.__unpack_data_package(payload, self.__output_config[payload[0]])
        else:
            _log.error('Unknown package command: ' + str(cmd))

    def __sendAndReceive(self, cmd, payload=b''):
        if self.__sendall(cmd, payload):
            return self.__recv(cmd)
        else:
            return None

    def __sendall(self, command, payload=b''):
        fmt = '>HB'
        size = struct.calcsize(fmt) + len(payload)
        buf = struct.pack(fmt, size, command) + payload

        if self.__sock is None:
            _log.error('Unable to send: not connected to Robot')
            return False

        _, writable, _ = select.select([], [self.__sock], [], DEFAULT_TIMEOUT)
        if len(writable):
            self.__sock.sendall(buf)
            return True
        else:
            self.__trigger_disconnected()
            return False

    def has_data(self):
        timeout = 0
        readable, _, _ = select.select([self.__sock], [], [], timeout)
        return len(readable)!=0

    def __recv(self, command, binary=False):
        while self.is_connected():
            try:
                self.__recv_to_buffer(DEFAULT_TIMEOUT)
            except RTSITimeoutException:
                return None

            # unpack_from requires a buffer of at least 3 bytes
            while len(self.__buf) >= 3:
                # Attempts to extract a packet
                packet_header = serialize.ControlHeader.unpack(self.__buf)

                if len(self.__buf) >= packet_header.size:
                    packet, self.__buf = self.__buf[3:packet_header.size], self.__buf[packet_header.size:]
                    data = self.__on_packet(packet_header.command, packet)
                    if len(self.__buf) >= 3 and command == Command.RTSI_DATA_PACKAGE:
                        next_packet_header = serialize.ControlHeader.unpack(self.__buf)
                        if next_packet_header.command == command:
                            _log.debug('skipping package(1)')
                            self.__skipped_package_count += 1
                            continue
                    if packet_header.command == command:
                        if(binary):
                            return packet[1:]

                        return data
                    else:
                        _log.debug('skipping package(2)')
                else:
                    break
        raise RTSIException(' _recv() Connection lost ')

    def __recv_to_buffer(self, timeout):
        readable, _, xlist = select.select([self.__sock], [], [self.__sock], timeout)
        if len(readable):
            more = self.__sock.recv(4096)
            #When the controller stops while the script is running
            if len(more) == 0:
                _log.error('received 0 bytes from Controller, probable cause: Controller has stopped')
                self.__trigger_disconnected()  
                raise RTSIException('received 0 bytes from Controller')

            self.__buf = self.__buf + more
            return True

        if (len(xlist) or len(readable) == 0) and timeout != 0: # Effectively a timeout of timeout seconds
            _log.warning('no data received in last %d seconds ',timeout)
            raise RTSITimeoutException("no data received within timeout")

        return False


    def __recv_from_buffer(self, command, binary=False):
        # unpack_from requires a buffer of at least 3 bytes
        while len(self.__buf) >= 3:
            # Attempts to extract a packet
            packet_header = serialize.ControlHeader.unpack(self.__buf)

            if len(self.__buf) >= packet_header.size:
                packet, self.__buf = self.__buf[3:packet_header.size], self.__buf[packet_header.size:]
                data = self.__on_packet(packet_header.command, packet)
                if packet_header.command == command:
                    if(binary):
                        return packet[1:]

                    return data
                else:
                    print('skipping package(2)')
            else:
                return None

    def __trigger_disconnected(self):
        _log.info("RTSI disconnected")
        self.disconnect() #clean-up

    def __unpack_protocol_version_package(self, payload):
        if len(payload) != 1:
            _log.error('RTSI_REQUEST_PROTOCOL_VERSION: Wrong payload size')
            return None
        result = serialize.ReturnValue.unpack(payload)
        return result.success

    def __unpack_elitecontrol_version_package(self, payload):
        if len(payload) != 16:
            _log.error('RTSI_GET_ELITECONTROL_VERSION: Wrong payload size')
            return None
        version = serialize.ControlVersion.unpack(payload)
        return version

    def __unpack_text_message(self, payload):
        if len(payload) < 1:
            _log.error('RTSIE_TEXT_MESSAGE: No payload')
            return None
        if(self.__protocolVersion == RTSI_PROTOCOL_VERSION_1):
            msg = serialize.MessageV1.unpack(payload)
        else:
            msg = serialize.Message.unpack(payload)

        if(msg.level == serialize.Message.EXCEPTION_MESSAGE or
           msg.level == serialize.Message.ERROR_MESSAGE):
            _log.error(msg.source + ': ' + msg.message)
        elif msg.level == serialize.Message.WARNING_MESSAGE:
            _log.warning(msg.source + ': ' + msg.message)
        elif msg.level == serialize.Message.INFO_MESSAGE:
            _log.info(msg.source + ': ' + msg.message)

    def __unpack_setup_outputs_package(self, payload):
        if len(payload) < 1:
            _log.error('RTSI_CONTROL_PACKAGE_SETUP_OUTPUTS: No payload')
            return None
        output_config = serialize.DataConfig.unpack_recipe(payload)
        return output_config

    def __unpack_setup_inputs_package(self, payload):
        if len(payload) < 1:
            _log.error('RTSI_CONTROL_PACKAGE_SETUP_INPUTS: No payload')
            return None
        input_config = serialize.DataConfig.unpack_recipe(payload)
        return input_config

    def __unpack_start_package(self, payload):
        if len(payload) != 1:
            _log.error('RTSI_CONTROL_PACKAGE_START: Wrong payload size')
            return None
        result = serialize.ReturnValue.unpack(payload)
        return result.success

    def __unpack_pause_package(self, payload):
        if len(payload) != 1:
            _log.error('RTSI_CONTROL_PACKAGE_PAUSE: Wrong payload size')
            return None
        result = serialize.ReturnValue.unpack(payload)
        return result.success

    def __unpack_data_package(self, payload, output_config):
        if output_config is None:
            _log.error('RTSI_DATA_PACKAGE: Missing output configuration')
            return None
        output = output_config.unpack(payload)
        return output

    def __list_equals(self, l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len((l1))):
            if l1[i] != l2[i]:
                return False
        return True
    
    @property
    def skipped_package_count(self):
        """The skipped package count, resets on connect"""
        return self.__skipped_package_count
