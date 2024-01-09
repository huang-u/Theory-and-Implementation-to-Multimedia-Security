import LSB as lsb
import AES as Cipher
import os


def main():
    print("Script started")
    select = input("Enter E for Encoding D for Decoding :")
    if select == 'E':
        if os.path.exists("Improved-LSB-Steganography/out.txt"):
            os.remove("Improved-LSB-Steganography/out.txt")
        if os.path.exists("pls.txt.enc"):
            os.remove("pls.txt.enc")
        if os.path.exists("pls.txt"):
            os.remove("pls.txt")
        if os.path.exists("Improved-LSB-Steganography/images/out1.png"):
            os.remove("Improved-LSB-Steganography/images/out1.png")

        if os.path.exists("Improved-LSB-Steganography/images/in1.png"):
            secretMessage = input("Enter the secret message :")
            passwordText = input("Password :")
            encodedMessage = Cipher.encrypt(secretMessage, passwordText)
            print(encodedMessage)
            lsb.LsbEncoding(encodedMessage)
            if os.path.exists("pls.txt"):
                os.remove("pls.txt")
        else : print("Image is not Present")



    if select == 'D':
        if os.path.exists("pls.txt.enc"):
            decodedText = lsb.LsbDecoding()
            print(decodedText)
            password = input("Enter the password :")
            finalMessage = Cipher.decrypt(decodedText, password)
            print("Final message :", finalMessage)
        else :
            print("PLS file is not present !")








main()
