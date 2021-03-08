.. image:: amdblack.jpg


===============================
Mesa Multimedia Installation
===============================

Prerequisites
--------------

- Ensure you have ROCm installed on the system. 

For ROCm installation instructions, see 

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html



System Prerequisites
#######################

The following operating systems are supported for Mesa Multimedia:

- Ubuntu 18.04.3 

- Ubuntu 20.04, including dual kernel 


Installation Prerequisites
############################ 
     
    
1. Obtain the AMDGPU driver from https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-20-45 for the appropriate distro version.



2. Follow the pre-installation instructions at https://amdgpu-install.readthedocs.io/en/latest/ (from “Preamble” to “Using the amdgpu-install Script” sections).



3. Proceed with the installation instructions as documented in the next section. 


    
Installation Instructions
----------------------------

1. Use the following installation instructions to install Mesa Multimeda:


:: 
     
      | ./amdgpu-install -y --no-dkms
      
      
.. note:: 

      Run it from the directory where the download is unpacked. The download and install instructions are:

     | $ cd ~/Downloads 
     | $ tar -Jxvf amdgpu-pro-YY.XX-NNNNNN.tar.xz
     | $ cd ~/Downloads/amdgpu-pro-YY.XX-NNNNNN
     | $ ./amdgpu-install -y --no-dkms     





2. ``gstreamer`` Installation


:: 
    
    sudo apt-get -y install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-      vaapi gstreamer1.0-libav gstreamer1.0-tools
    
    sudo apt-get -y install gst-omx-listcomponents gstreamer1.0-omx-bellagio-config gstreamer1.0-omx-generic gstreamer1.0-omx-generic-config

   

    
3. Utilities Installation

:: 
    
     sudo apt-get -y install mediainfo ffmpeg

     sudo reboot
     
     # Check amdgpu loadking status after reboot

     dmesg | grep -i initialized

     Sep 24 13:00:42 jz-tester kernel: [  277.120055] [drm] VCN decode and encode initialized successfully.

     Sep 24 13:00:42 jz-tester kernel: [  277.121654] [drm] Initialized amdgpu 3.34.0 20150101 for 0000:03:00.0 on minor 1
    



4. Configure Running Environment Variables

:: 

     export BELLAGIO_SEARCH_PATH=/opt/amdgpu/lib/x86_64-linux-gnu/libomxil-bellagio0:/opt/amdgpu/lib/libomxil-bellagio0

     export GST_PLUGIN_PATH=/opt/amdgpu/lib/x86_64-linux-gnu/gstreamer-1.0/

     export GST_VAAPI_ALL_DRIVERS=1

     export OMX_RENDER_NODE=/dev/dri/renderD128
    
   
    



Check Installation 
--------------------

1. Ensure you perform an installation check. 

   
::  

     omxregister-bellagio -v

     Scanning directory /opt/amdgpu/lib/libomxil-bellagio0/

     Scanning library /opt/amdgpu/lib/libomxil-bellagio0/libomx_mesa.so

     Component OMX.mesa.video_decoder registered with 0 quality levels

     Specific role OMX.mesa.video_decoder.mpeg2 registered

     Specific role OMX.mesa.video_decoder.avc registered

     Specific role OMX.mesa.video_decoder.hevc registered

     Component OMX.mesa.video_encoder registered with 0 quality levels

     Specific role OMX.mesa.video_encoder.avc registered

 
     2  OpenMAX IL ST static components in 1 library successfully scanned


::        
     
     gst-inspect-1.0 omx
        




Plugin Details

    +---------------------------------------+--------------------------------------+
    | Name                                  | OMX                                  |                                                    
    +---------------------------------------+--------------------------------------+
    | Description                           | GStreamer OpenMAX Plug-ins           |
    +---------------------------------------+--------------------------------------+
    | Filename                              | /usr/lib/x86_64-linux-gnu/           |
    |                                       | gstreamer-1.0/libgstomx.so           |
    +---------------------------------------+--------------------------------------+
    | Version                               | 1.12.4                               |
    +---------------------------------------+--------------------------------------+
    | License                               |  LGPL                                |
    +---------------------------------------+--------------------------------------+
    | Source module                         | gst-omx                              |
    +---------------------------------------+--------------------------------------+
    | Source release date                   |  2017-12-07                          |
    +---------------------------------------+--------------------------------------+
    | Binary package                        | GStreamer OpenMAX Plug-ins source    |
    |                                       | release                              |
    +---------------------------------------+--------------------------------------+
    | Origin URL                            |  Unknown package origin              |
    +---------------------------------------+--------------------------------------+                    



::        

 
     omxmpeg2dec: OpenMAX MPEG2 Video Decoder
     
     omxh264dec: OpenMAX H.264 Video Decoder
     
     omxh264enc: OpenMAX H.264 Video Encoder 

 
     3. Features
 
     +-- 3 elements 
     
::    
  
        gst-inspect-1.0 vaapi
    
     
  
  
  
 
   
Plugin Details
  
    +---------------------------------------+--------------------------------------+
    | Name                                  | vaapi                                |                                                    
    +---------------------------------------+--------------------------------------+
    | Description                           | VA-API based elements                |
    +---------------------------------------+--------------------------------------+
    | Filename                              | /usr/lib/x86_64-linux-gnu/           |
    |                                       | gstreamer-1.0/libgstvaapi.so         | 
    +---------------------------------------+--------------------------------------+
    | Version                               | 1.14.5                               |
    +---------------------------------------+--------------------------------------+
    | License                               |  LGPL                                |
    +---------------------------------------+--------------------------------------+
    | Source module                         | gstreamer-vaapi                      |
    +---------------------------------------+--------------------------------------+
    | Source release date                   |  2019-05-29                          |
    +---------------------------------------+--------------------------------------+
    | Binary package                        | gstreamer-vaapi                      |
    |                                       |                                      |
    +---------------------------------------+--------------------------------------+
    | Origin URL                            |                                      |
    |                                       | http://bugzilla.gnome.org            |
    |                                       |/enter_bug.cgi?product=GStreamer      |
    +---------------------------------------+--------------------------------------+                    
                   




   

::

      vaapijpegdec: VA-API JPEG decoder
      vaapimpeg2dec: VA-API MPEG2 decoder
      vaapih264dec: VA-API H264 decoder
      vaapivc1dec: VA-API VC1 decoder
      vaapivp9dec: VA-API VP9 decoder
      vaapih265dec: VA-API H265 decoder
      vaapipostproc: VA-API video postprocessing
      vaapidecodebin: VA-API Decode Bin
      vaapisink: VA-API sink
      vaapih265enc: VA-API H265 encoder
      vaapih264enc: VA-API H264 encoder

    11 Features
   
    +-- 11 elements
    
 



