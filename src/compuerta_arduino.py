"""Conexión serial sencilla con la maqueta de compuerta en Arduino para ReCycle Vision.

Guardar como src/compuerta_arduino.py.
Requiere: pip install pyserial
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def listar_puertos_com() -> list[str]:
    if list_ports is None:
        return []
    return [p.device for p in list_ports.comports()]


class CompuertaArduino:
    """Envía 'ABRIR' por puerto serial cuando la decisión es ACEPTADO."""

    def __init__(
        self,
        puerto: Optional[str] = None,
        baudrate: int = 9600,
        cooldown_seg: float = 2.5,
    ) -> None:
        env_puerto = os.environ.get("RECYCLE_ARDUINO_PORT", "").strip()
        self.puerto: Optional[str] = puerto or (env_puerto if env_puerto else None)
        self.baudrate = baudrate
        self.cooldown_seg = cooldown_seg
        self._ser = None
        self._ultima_apertura: float = 0.0

    def conectar(self, puerto: Optional[str] = None) -> bool:
        if puerto is not None:
            self.puerto = puerto
        self.cerrar()

        if not self.puerto:
            puertos = listar_puertos_com()
            if puertos:
                self.puerto = puertos[0]
            else:
                print("[Arduino] Sin puerto COM detectado (modo simulación activo).")
                return False

        if serial is None:
            print("[Arduino] 'pyserial' no instalado (instala con: pip install pyserial).")
            return False

        try:
            self._ser = serial.Serial(self.puerto, self.baudrate, timeout=1)
            time.sleep(1.8)
            self._ser.reset_input_buffer()
            print(f"[Arduino] Conectado en {self.puerto} ({self.baudrate} baudios).")
            return True
        except Exception as exc:
            print(f"[Arduino] No se pudo abrir {self.puerto}: {exc} (modo simulación activo).")
            self._ser = None
            return False

    @property
    def conectada(self) -> bool:
        return bool(self._ser is not None and self._ser.is_open)

    def notificar_decision(self, decision: str) -> bool:
        """Abre la compuerta solo si decision == 'ACEPTADO'."""
        if decision != "ACEPTADO":
            return False

        ahora = time.monotonic()
        if (ahora - self._ultima_apertura) < self.cooldown_seg:
            return False

        self._ultima_apertura = ahora
        if self.conectada and self._ser is not None:
            try:
                self._ser.write(b"ABRIR\n")
                self._ser.flush()
                print(f"[Arduino -> {self.puerto}] ABRIR")
                return True
            except Exception as exc:
                print(f"[Arduino] Error al enviar ABRIR: {exc}")
                return False

        print("[Arduino SIMULACIÓN] ABRIR (0° -> 70° -> 0°)")
        return True

    def cerrar(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.write(b"CERRAR\n")
                    self._ser.close()
            except Exception:
                pass
            self._ser = None