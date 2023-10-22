from kivymd.uix.screen import MDScreen
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivymd.uix.button import MDFillRoundFlatIconButton ,MDRoundFlatButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
# from kivymd.uix.toolbar import MDToolbar
from kivymd.uix.toolbar import MDTopAppBar



class ConvertApp(MDApp):

    def flip(self):
        if self.state == 0:
            self.state = 1
            self.toolbar.title = "Decimal to Binary"
            self.input.text = "Enter a decimal number"
            self.converted.text = ""
            self.lable.text = ""
        else:
            self.state = 0
            self.toolbar.title = "Binary to Decimal"
            self.input.text = "Enter a Binary number"
            self.converted.text = ""
            self.lable.text = ""


    def convert(self, args):
        if self.state == 0:
            val = int(self.input.text, 2)
            self.converted.text = str(val)
            self.lable.text = "in decimal is:"
        else:
            val = bin(int(self.input.text))[2:] #bin(int(8)) = 0b1000
            self.converted.text = val
            self.lable.text = "in binary is:"


    def build(self):
        self.state = 0
        #self.theme_cls.primary_palette = "DeepBlu" #DeepPurple
        screen = MDScreen()


        # self.toolbar = MDToolbar(title="Binary to Decimal")
        self.toolbar = MDTopAppBar(title="Binary to Decimal")
        self.toolbar.pos_hint = {"top" : 1}
        self.toolbar.right_action_items = [
        ["rotate-3d-variant", lambda x: self.flip()]]
        screen.add_widget(self.toolbar)

        #logo
        screen.add_widget(Image(
            source='Converters.png',
            pos_hint={"center_x":0.50, "center_y":0.7},
            size_hint_x=0.45,
            size_hint_y=0.6))

        #collect user input
        self.input = MDTextField(
            text="enter a binary number",
            halign="center",
            size_hint=(0.8, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.45},
            font_size=22
        )

        screen.add_widget(self.input)

        #secondary + primary labels
        self.lable = MDLabel(
        halign = "center",
        pos_hint = {"center_x": 0.5, "center_y": 0.35},
        theme_text_color="Secondary"
        )

        self.converted = MDLabel(
            halign="center",
            pos_hint={"center_x": 0.5, "center_y": 0.3},
            theme_text_color="Primary",
            font_style="H5"
        )

        screen.add_widget(self.lable)
        screen.add_widget(self.converted)

        #convert button
        screen.add_widget(MDFillRoundFlatIconButton(
            text="CONVERT",
            font_size=17,
            pos_hint={"center_x": 0.5, "center_y": 0.15},
            on_press=self.convert,
            #icon="language-python"
            icon="hand-pointing-right" ## Add hand icon
        ))

        return screen

#if __name__ = '__main__' :
ConvertApp().run()



