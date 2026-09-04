#!/usr/bin/env python3
"""Normalize Unitree legacy CycloneDDS XML for Ubuntu 24.04."""
from pathlib import Path
import sys

target = Path(sys.argv[1])
original = target.read_text()
start = original.index("ChannelConfigHasInterface =")
end = original.index("ChannelConfigAutoDetermine =")
replacement = '''ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="$__IF_NAME__$" multicast="false"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
      <DontRoute>true</DontRoute>
    </General>
  </Domain>
</CycloneDDS>"""

'''
target.write_text(original[:start] + replacement + original[end:])
