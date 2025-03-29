
print("📷 Handwritten Kyrgyz Letter Recognition")
print("1 — Запустить веб-камеру")
print("Другое — Выйти")

choice = input("Введите номер: ")

if choice == '1':
    import webcam_predict
else:
    print("Выход из программы.")
