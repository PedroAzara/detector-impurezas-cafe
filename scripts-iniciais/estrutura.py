classes = ['1T', '2T', '3T', '4T', '5T', '6T', '7T', 'mao']
prefixos = ['cafe', 'cafe_Luisa', 'milho1', 'milho3']

with open("estrutura_dataset.txt", "w") as f:
    for classe in classes:
        f.write(f"{classe}/\n")
        for prefixo in prefixos:
            for i in range(1, 15):
                nome = f"{prefixo}_{i:02d}_{classe}.png"
                f.write(f"├── {nome}\n")
        f.write("\n")
