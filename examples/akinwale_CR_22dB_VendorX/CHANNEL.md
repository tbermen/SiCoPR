# The channel this example runs on

The S-parameter files are an IEEE 802.3 contribution. They are public, and they are not ours to redistribute, so this directory names them instead of shipping them.

## Where to get them

1. Download **https://www.ieee802.org/3/dj/public/tools/CR/akinwale_3dj_02_2311.zip**

   (listed as *212 Gb/s Per Lane PAM4 CR Channels with Flexible Host Architectures and Longer Reach Cables - NIC Perspective*, 27 Nov 2023, on the IEEE P802.3dj channel and tool page: https://www.ieee802.org/3/dj/public/tools/index.html).

2. Unpack it. The six files this example uses are in

   `akinwale_3dj_02_2311/akinwale_3dj_01_2311/0_22dB_VendorX/`.

3. Check you have the same files. SHA-256:

```
4e7d787f89a4180b4c3c9cd830309556d7c0332e2cab629f393285316a515e28  Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_thru1.s4p
50ddc0ed73bcd519cd6285af112c3543336c1cf944e68e7e74c57c4b12d0f776  Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk1_Fext.s4p
4daab46464437242d32a95394de9de1c72a81b34736f2945512eae61768697fc  Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk2_Fext.s4p
8dfee958f31aa5b1eb5622be14c2be447942d12fcd5b335568d53087f7bc208d  Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk3_Fext.s4p
d9ae37afaa927cb0d3f1e3afd852cbd5c1c92d6b43c5bf5b8a1faca2cc53d5aa  Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk4_Next.s4p
6617f4cd77c9bfbb2e8e5a18d1986c759bcce509264c5fa89b4b82f2676fb366  Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk5_Next.s4p
```

| file | role | size |
|---|---|---|
| `Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_thru1.s4p` | THRU | 3.8 MB |
| `Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk1_Fext.s4p` | FEXT | 3.9 MB |
| `Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk2_Fext.s4p` | FEXT | 4.0 MB |
| `Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk3_Fext.s4p` | FEXT | 4.0 MB |
| `Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk4_Next.s4p` | NEXT | 3.8 MB |
| `Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_xtalk5_Next.s4p` | NEXT | 3.8 MB |

If a checksum does not match, the file is a different revision of the contribution and the expected values in this directory will not reproduce. That is worth knowing before you conclude anything about the engine.

