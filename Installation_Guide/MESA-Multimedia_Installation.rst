.. image:: amdblack.jpg


===============================
MESA Multimedia Installation
===============================

Prerequisites
--------------

- Ensure you have ROCm installed on the system. 

For ROCm installation instructions, see 

https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html


MESA Multimedia Installation
-------------------------------

System Prerequisites
#######################

The following operating systems are supported for Mesa Multimedia:

- Ubuntu 18.04.3 

- Ubuntu 20.04, including dual kernel 


.. note::

    Ensure Mesa Multimedia is a fresh and clean installation. Any previously installed versions of AMD ROCm must be uninstalled before installing Mesa Multimedia.
  
 
Installation Prerequisites
############################
 
1. Use the following instructions to ensure the system on which you want to install Mesa Multimedia is up-to-date:

::

    sudo apt update

    sudo apt dist-upgrade

    

2. Select the desired repository package to download the amdgpu graphics stack packages based on your required Ubuntu version and branch of code. 

    +---------------------------------------+--------------------------------------+
    | Ubuntu 18.04                          | Ubuntu 20.04                         |                                                    
    +=======================================+======================================+
    | amd-nonfree-mainline_18.04-1_all.deb  | amd-nonfree-mainline_20.04-1_all.deb |
    +---------------------------------------+--------------------------------------+
    | amd-nonfree-VERSION_18.04-1_all.deb   | amd-nonfree-VERSION_20.04-1_all.deb  |
    +---------------------------------------+--------------------------------------+
    | amd-nonfree-staging_18.04-1_all.deb   | amd-nonfree-staging_20.04-1_all.deb  |
    +---------------------------------------+--------------------------------------+



.. note::

  For installing release drivers, VERSION must be replaced with a driver version. For example,  19.40, 19.50, 20.10, and others.
    
    
 3.	Use the following instructions to download and install the selected package:
 
 ::

   MIRROR=artifactory-cdn.amd.com/artifactory/list/amdgpu-deb

   REPO_PKG=amd-nonfree-mainline_18.04-1_all.deb

   cd /tmp

   wget http://${MIRROR}/${REPO_PKG}

   sudo dpkg -i ${REPO_PKG} 

    
Installation Instructions
##########################

1. Use the following installation instructions to install MESA Multimeda:

:: 
    
    sudo apt install -y ./amd-nonfree-mainline_20.04-1_all.deb && sudo apt update
    
    sudo amdgpu-install -y --no-dkms

2. gstreamer Installation

:: 
    
    sudo apt-get -y install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-vaapi         gstreamer1.0-libav gstreamer1.0-tools
    
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
    
5. Configure Running Environment Variables

:: 

    export BELLAGIO_SEARCH_PATH=/opt/amdgpu/lib/x86_64-linux-gnu/libomxil-bellagio0:/opt/amdgpu/lib/libomxil-bellagio0
    
    export GST_PLUGIN_PATH=/opt/amdgpu/lib/x86_64-linux-gnu/gstreamer-1.0/
    
    export GST_VAAPI_ALL_DRIVERS=1
    
    export OMX_RENDER_NODE=/dev/dri/renderD128


Check Installation 
##########################

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
    
    
 
 2 OpenMAX IL ST static components in 1 libraries successfully scanned

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
 
 ::
 
     +-- 3 elements 
     
    gst-inspect-1.0 vaapi

Plugin Details

  Name                     vaapi
  Description              VA-API based elements
  Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstvaapi.so
  Version                  1.14.5
  License                  LGPL
  Source module            gstreamer-vaapi
  Source release date      2019-05-29
  Binary package           gstreamer-vaapi
  Origin URL               http://bugzilla.gnome.org/enter_bug.cgi?product=GStreamer
 
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
 
  11 features
  
 :: 
    
    +-- 11 elements
    

Verification Test
###################

MPEG2 Decode
**************
::

    gst-launch-1.0 -f filesrc location=./mpeg2/1080p/hdwatermellon_1_5.mpg ! mpegpsdemux ! mpegvideoparse ! omxmpeg2dec ! filesink location=t.yuv

    gst-launch-1.0 -f filesrc location=./mpeg2/1080p/hdwatermellon_1_5.mpg ! mpegpsdemux ! mpegvideoparse ! vaapimpeg2dec ! filesink location=t.yuv

    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./mpeg2/1080p/hdwatermellon_1_5.mpg  -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv


AVC/H264 Decode
****************
::

    gst-launch-1.0 filesrc location=./1080p_H264.mp4 ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! omxh264dec ! filesink location=t.yuv

    gst-launch-1.0 filesrc location=./1080p_H264.mp4 ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! vaapih264dec ! filesink location=t.yuv

    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./1080p_H264.mp4  -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv

    gst-launch-1.0 filesrc location=./h264/4k/4K-CHIMEI-INN-60MBPS.MP4 ! qtdemux name=demux demux.video_0 ! queue ! h264parse ! vaapih264dec ! filesink location=t.yuv

    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./h264/4k/4K-CHIMEI-INN-60MBPS.MP4  -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv


AVC/H264 Encode
****************
::

    gst-launch-1.0 -f videotestsrc num-buffers=100 ! omxh264enc ! filesink location=t.h264

    gst-launch-1.0 -f videotestsrc num-buffers=100 ! vaapih264enc ! filesink location=t.h264

    ffmpeg  -vaapi_device /dev/dri/renderD129  -s 1920x1080 -pix_fmt yuv420p -i t.yuv -vf 'format=nv12|vaapi,hwupload' -c:v h264_vaapi   out.mp4


VC1 Decode
**********
::

    gst-launch-1.0 -v filesrc location=./vc1/1080p/1080P_ElephantsDream.wmv ! asfdemux ! vaapivc1dec ! filesink location=t.yuv

    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./vc1/1080p/1080P_ElephantsDream.wmv  -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv


HEVC/H265 decode
*****************
::

    gst-launch-1.0 filesrc location=./h265/Guardians_of_the_galaxy_trailer_720p.mp4 ! qtdemux name=demux demux.video_0 ! queue ! h265parse ! vaapih265dec ! filesink    location=t.yuv

    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./h265/Guardians_of_the_galaxy_trailer_720p.mp4  -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv

    #10Bit
    ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD129 -i ./Perfume_1080p_h265_10bit.mp4 -vcodec rawvideo -pixel_format yuv420p ./t.yuv


HEVC/H265 encode
******************
::

    gst-launch-1.0 -f videotestsrc num-buffers=100 ! vaapih265enc ! filesink location=t.h265
    
    ffmpeg  -vaapi_device /dev/dri/renderD129  -s 1920x1080 -pix_fmt yuv420p -i t.yuv -vf 'format=nv12|vaapi,hwupload' -c:v hevc_vaapi   out.mp4


VP9 decode
******************
::

    gst-launch-1.0 filesrc location=./VP9/'Grubby Grubby vs Neytpoh Pt.1 Warcraft 3 ORC vs NE Twisted -1.webm' ! matroskademux ! vaapivp9dec ! filesink location=t.yuv
 
    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./VP9/'Grubby Grubby vs Neytpoh Pt.1 Warcraft 3 ORC vs NE Twisted -1.webm'  -bf 0  -c:v rawvideo -    pix_fmt yuv420p t.yuv
 
    #10Bit
    ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD129 -i ./crowd_run_4096X2176_fr30_bd10_4buf_l5.webm -vcodec rawvideo -pixel_format yuv420p ./t.yuv


MJPEG Decode
******************
::

    gst-launch-1.0 filesrc location=./MJPEG/004_motion_720p60-420-lq.avi ! jpegparse ! vaapijpegdec ! filesink location=t.yuv
 
    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./MJPEG/004_motion_720p60-420-lq.avi -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv


VC1 Decode
******************
::

    gst-launch-1.0 -v filesrc location=./vc1/1080p/1080P_ElephantsDream.wmv ! asfdemux ! vaapivc1dec ! filesink location=t.yuv
 
    ffmpeg -hwaccel vaapi  -hwaccel_device /dev/dri/renderD129 -i ./vc1/1080p/1080P_ElephantsDream.wmv -bf 0  -c:v rawvideo -pix_fmt yuv420p t.yuv


Transcode
******************
::

    gst-launch-1.0 -f filesrc location=./h264/1080p/Inception.mp4 ! qtdemux ! vaapih264dec ! vaapih265enc ! filesink location=t.h265

    gst-launch-1.0 -f filesrc location=./h265/shaun_white_480p.mp4 ! qtdemux ! vaapih265dec ! vaapih264enc ! filesink location=t.h264

    ffmpeg -hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD129 -i mpeg2/1080p/hdwatermellon_1_5.mpg  -bf 0 -c:v h264_vaapi ~/output.mp4

    ffmpeg -hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD129 -i mpeg2/1080p/hdwatermellon_1_5.mpg  -bf 0 -c:v hevc_vaapi ~/output.mp4


Notes
=========

1.	MI100 has no X server up, so decode image will be dumped into a YUV (NV12) format file. it can be offline checked with YUV player

2.	mediainfo can help you detect original clip's format and resolution.(e.g. mediainfo  ./MJPEG/004_motion_720p60-420-lq.avi)

3.	ffmpeg can be used to play YUV image file. (e.g. ffplay -framerate 30 -f rawvideo -video_size 1920x1080 -pixel_format nv12 t.yuv )

4.	For VAAPI decode, output video size needs 16-alignment, eg. 1920x1080 after decode, 1920x1088 needs be used to play.

5.	You can find a quick test script in attachment. You need download the mm_test_arct.instr also. the test clip is located: http://lnx-jfrog/artifactory/linux-ci-generic-local/mesa/1080p_H264.mp4

6.	vooya :: raw Video Sequence Player:  https://www.offminor.de/

7.	the below command can list the available amdgpu device render nodes:

::

    for i in $(ls /dev/dri/renderD* | xargs -l basename | cut -c8-);do [[ "$(grep "amdgpu" /sys/kernel/debug/dri/$i/name)" == "" ]] && continue;echo    "AMD_RENDER_NODE=/dev/dri/renderD$i";done

