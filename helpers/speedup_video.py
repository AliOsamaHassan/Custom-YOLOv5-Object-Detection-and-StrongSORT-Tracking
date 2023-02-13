from moviepy.editor import VideoFileClip, vfx

in_loc = 'test_inference_video.mp4'
out_loc = in_loc.replace(".mp4", "_FASTER.mp4")

# Import video clip
clip = VideoFileClip(in_loc)
print("fps: {}".format(clip.fps))

# Apply speed up
final = clip.fx(vfx.speedx, 2)
print("fps: {}".format(final.fps))

# Save video clip
final.write_videofile(out_loc)
# final.write_gif(out_loc)
