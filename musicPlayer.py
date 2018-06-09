import pyglet
import webbrowser
import threading
import constants
import os
import random
import winsound
import time
MUSIC_SAVE_DIRECTORY = './music'
MUSIC_EMOTION_DIRECTPRY = [MUSIC_SAVE_DIRECTORY + '/' + i for i in constants.EMOTIONS]
MUSIC_LIST = dict()
for i in range(len(constants.EMOTIONS)):
    if not os.path.exists(MUSIC_SAVE_DIRECTORY):
        os.makedirs(MUSIC_SAVE_DIRECTORY)
    if not os.path.exists(MUSIC_EMOTION_DIRECTPRY[i]):
        os.makedirs(MUSIC_EMOTION_DIRECTPRY[i])
        continue
    MUSIC_LIST[i] = os.listdir(MUSIC_EMOTION_DIRECTPRY[i])
class musicPlayer(threading.Thread):
    def emotionMark(self):
        return self.emotionMark
    def isPlaying(self):
        if self.player._get_source() == None:
            return False
        return self.player._playing
    def __init__(self):
        self.player = pyglet.media.Player()
        self.emotionMark = -1
        # self.source = None
    def load(self, emotion):
        self.emotionMark = emotion
        path = MUSIC_EMOTION_DIRECTPRY[emotion]+'/'+MUSIC_LIST[emotion][random.randint(a=0, b=len(MUSIC_LIST[emotion])-1)]
        print(path)
        source = pyglet.media.load(path)
        if self.player._get_source() != None:
            if self.player._playing:
                self.stop()
            self.player._groups[0] = source
        else:
            self.player.queue(source)
        print('self.musc = SND_PURGE')
        # webbrowser.open(path)
        print('load() after')
        # pygame.mixer.music.set_volume(1.0)
    def run(self):
        print('self.musc = SND_ASYNC before')
        if self.player.source != None:
            self.player.play()
        print('self.musc = SND_ASYNC after')
    def pause(self):
        if self.player.source != None and self.player._playing:
            self.player.pause()
    def resume(self):
        print('self.musc = SND_NOSTOP before')
        if self.player.source != None and self.player._playing != None:
            self.player.play()
        print('self.musc = SND_NOSTOP after')
    def stop(self):
        print('self.musc = SND_PURGE before')
        if len(self.player._groups) > 0: # there is a music source has been loaded
            self.pause() # pause
            self.player._groups.clear() # remove the source
            self.player.delete() #Tear down the player and any child objects.
            print(len(self.player._groups))
        # winsound.PlaySound(self.music, winsound.SND_PURGE)
        print('self.musc = SND_PURGE after')
# class musicPlayer():
#     def __init__(self):
#         pygame.mixer.pre_init(44100, -16, 2, 2048) # setup mixer to avoid sound lag
#         pygame.init()
#         self.screen = pygame.display.set_mode((10,10))
#         pygame.display.flip()
#     def load(self, emotion):
#         path = MUSIC_EMOTION_DIRECTPRY[emotion]+'/'+MUSIC_LIST[emotion][random.randint(a=0, b=len(MUSIC_LIST[emotion])-1)]
#         print(path)
#         self.track = pygame.mixer.music.load(path) # 0 ~ len()-1
#         self.music = pygame.mixer.music
#         # pygame.mixer.music.set_volume(1.0)
#         self.music.set_volume(1.0)
#     def run(self):
#         self.music.play()
#         while self.music.get_busy():
#             print('playing')
#             pygame.time.Clock().tick(10)# important code for playing
#     def pause(self):
#         pygame.mixer.music.pause()
#     def resume(self):
#         pygame.mixer.music.unpause()
#     def stop(self):
#         pygame.mixer.music.stop()

# class musicPlayer(threading.Thread):
#     def __init__(self, *args, **kwargs):
#         super(musicPlayer, self).__init__(*args, **kwargs)
#         self.__flag = threading.Event() #用于暂停的flag
#         self.__llag.set() #set with True
#         self.__running = threading.Event() # used to stop the thread
#         self.__running.set()
#
#     def run(self):
#         pygame.init()
#         while self.__running.isSet():
#             self.__flag.wait() # If it is True, then return. Otherwise block the process until it is True
#             #do something and sleep
#     def pause(self):
#         self.__flag.clear() # set with False to block the process
#
#     def resume(self):
#         self.__flag.set()
#
#     def stop(self):
#         if not self.__flag.isSet():
#             self.__flag.set()
#         self.__running.clear()
if __name__ == '__main__':
    obj = musicPlayer()
    obj.load(3)
    obj.run()

