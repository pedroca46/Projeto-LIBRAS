import os
from utils import predictRT
from utils import trainNpyFile
from viewCam import opencam
from numpy import array
from time import time

from tensorflow.keras.models import load_model

DATA_PATH = os.path.join('MP_Data') 
no_sequences = 30
sequence_length = 30
threshold = 0.9

def makedirs(actions):
    DATA_PATH = os.path.join('MP_Data') 
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu(model, actions):
    while True:
        clear_screen()
        print("Menu do Projeto Libras")
        print("======================")
        print("1. Testar câmera")
        print("2. Coletar dados de treino")
        print("3. Predição em tempo real")
        print("4. Treinar a IA")
        print("5. Sair")
        print("======================")
        
        choice = input("Escolha uma opção: ")

        if choice == '1':
            testarCam()
        elif choice == '2':
            colectData()
        elif choice == '3':
            predict(model)
        
        elif choice == '5':
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")
            input("Pressione Enter para continuar...")


def altMenu(actions):
    while True:
        clear_screen()
        print("Menu do Projeto Libras")
        print("======================")
        print("1. Testar câmera")
        print("2. Coletar dados de treino")
        print("3. Treinar a IA")
        print("4. Carregar modelo")
        print("5. Sair")
        print("======================")
        
        choice = input("Escolha uma opção: ")

        if choice == '1':
            testarCam()
        elif choice == '2':
            colectData(actions)
        elif choice == '3':
            mtreinarIA()
        elif choice == '4':
            print("carregar modelo")
        elif choice == '5':
            print("Saindo...")
            break
        else:
            print("Opção inválida! Tente novamente.")
            input("Pressione Enter para continuar...")




def testarCam():
    clear_screen()
    print("                 === Abrir Câmera ===            ")
    print(" => Digite o índice do seu dispositivo (câmera), começa em 0:")
    inputcam = int(input(" => "))
    print("             === Abrindo sua câmera... ===            ")
    opencam(inputcam)
    input("Pressione Enter para continuar...")


def colectData(actions):
    clear_screen()
    print("         === Coleta de dados ===         ")
    print(" => Digite o índice do seu dispositivo (câmera), começa em 0:")
    inputcam = int(input(" => "))
    print("     Gestos para coletar => ", actions)
    input("     Pressione Enter para iniciar...")
    trainNpyFile(actions, inputcam)
    print("         === Coleta finalizada ===         ")
    input("Pressione Enter para continuar...")

def mtreinarIA():
    from utils import treinarIA
    clear_screen()
    print("         === Treinar a IA ===        ")
    print("\n   => Quantas épocas você deseja para o seu treino?")  
    epochs = int(input("   =: "))
    model = treinarIA(epochs, actions)
    print("\n   Agora, digite o nome do arquivo para salvá-lo.")
    nome_arq = str(input("   => "))
    model.save("Weights/{}.h5".format(nome_arq))
    print("\n         === Treinamento concluído e peso salvo! ===        ")
    time.sleep(2)
    clear_screen()
    print("Utilizaremos o peso salvo para dar continuidade, para usar outro, reinicie o script.")
    time.sleep(2)
    clear_screen()
    model = load_model('Weights/{}.h5'.format(nome_arq))
    
    return model

def predict(model):
    clear_screen()
    print("             === Predição em Tempo Real ===            ")
    print("=> Exibindo resultado...")
    predictRT(model, actions)
    print(type(model))
    input("=> Pressione Enter para continuar...")



if __name__ == "__main__":
    import time

    clear_screen()
    print("""
          ~===========================~
=> BEM-VINDO! Antes de começar a usar,
escolha o nome do arquivo .h5 (Arquivo de peso) que você deseja usar.
          obs: não digite a extensão
          ~===========================~

        """)
    model_path = str(input("=> Weights/"))
    if model_path.lower() == "sair":
        pass
    else:
        model = load_model(f"Weights/{model_path}.h5")
    clear_screen()

    actions = []

    clear_screen()
    while True:
        gesture = input("Digite um gesto (somente letras e espaços, ou 'sair' para finalizar): ")
        if gesture.lower() == 'sair':
            break
        if gesture.replace(" ", "").isalpha() and not (gesture.startswith(" ") or gesture.endswith(" ")):
            actions.append(gesture)
        else:
            print("Entrada inválida. Por favor, digite apenas letras e espaços, sem espaços no início ou no fim.")

    clear_screen()
    #actions = array(['Tudo bem1', '_', 'por que', 'bom dia', 'Oi'])
    actions = array(actions)

    clear_screen()
    print("         === Criando diretórios ===          ")
    makedirs(actions)
    print("   => Diretórios criados!")
    time.sleep(1)
    clear_screen()

    try:
        menu(model, actions)
    except:
        import time
        print("Nenhum modelo foi carregado, a predição em tempo real será desativada.")
        time.sleep(3)
        clear_screen()
        altMenu(actions)

