Scripts que contienen el código necesario para desarrollar el ajuste con RL a una recta fija.

<ul>
  <li>env_linear_simple.py: Environment de ajuste con RL a una recta fija para todos los episodios.</li>
  <li>agents.py: Script con los Agentes de RL implementados que se han utilizado.</li>
  <li>training_Q.py: Script para entrenar desde la terminal el ajuste a la recta con QLearning.</li>
  <li>training_DQN.py: Script para entrenar desde la terminal el ajuste a la recta con DeepQLearning.</li>
  <li>run_saved_weights.ipynb: Notebook en el que se cargan los pesos de un agente ya entrenado (por ejemplo los pesos 
  en la carpeta saved_weights/) y permite ejecutar un bucle en el que se pone a prueba el ajuste por el agente.</li>
  <li>Figures/: Carpeta donde se almacenan las gráficas del entrenamiento de los agentes cuyos pesos están guardados en saved_weights/.</li>
</ul>
