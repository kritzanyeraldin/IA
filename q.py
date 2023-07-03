import pygame
import random
import math
import matplotlib.pyplot as plt


# Dimensiones de la ventana del juego
WIDTH = 800
HEIGHT = 600

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Tamaño y velocidad de la serpiente
SNAKE_SIZE = 20
SNAKE_SPEED = 35

# Inicialización de Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

q_table1 = {}
q_table2 = {}
threshold_distance = 10  # Distancia mínima para recibir una recompensa positiva
#prev_distance=99
performance = []
convergence = []


class SnakeGame:
    def __init__(self):
        self.snake = [(int(WIDTH/2), int(HEIGHT/2))]
        self.direction = 'left'
        self.food = self.generate_food()


    def generate_food(self):
        x = random.randrange(0, WIDTH - SNAKE_SIZE, SNAKE_SIZE)
        y = random.randrange(0, HEIGHT - SNAKE_SIZE, SNAKE_SIZE)
        return (300, 300)

    def draw_snake(self):
        for segment in self.snake:
            pygame.draw.rect(screen, GREEN, (segment[0], segment[1], SNAKE_SIZE, SNAKE_SIZE))

    def draw_food(self):
        pygame.draw.rect(screen, RED, (self.food[0], self.food[1], SNAKE_SIZE, SNAKE_SIZE))

    def move_snake(self):

        head = self.snake[0]
        x, y = head
        if self.direction == 'right':
            x += SNAKE_SIZE
        elif self.direction == 'left':
            x -= SNAKE_SIZE
        elif self.direction == 'up':
            y -= SNAKE_SIZE
        elif self.direction == 'down':
            y += SNAKE_SIZE
        new_head = (x, y)
        self.snake.insert(0, new_head)
        if self.snake[0] == self.food:
            self.food = self.generate_food()
        else:
            self.snake.pop()

    def check_collision(self):
        head = self.snake[0]
        if (
            head[0] < 0 or
            head[0] >= WIDTH or
            head[1] < 0 or
            head[1] >= HEIGHT or
            head in self.snake[1:]
        ):
            return True
        return False

    def update_score(self):
        font = pygame.font.Font(None, 36)
        score_text = font.render("Score: " + str(len(self.snake)), True, WHITE)
        screen.blit(score_text, (10, 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and self.direction != 'left':
                    self.direction = 'right'
                elif event.key == pygame.K_LEFT and self.direction != 'right':
                    self.direction = 'left'
                elif event.key == pygame.K_UP and self.direction != 'down':
                    self.direction = 'up'
                elif event.key == pygame.K_DOWN and self.direction != 'up':
                    self.direction = 'down'

    def set(self, direction):
        if direction == 'right' and self.direction != 'left':
            self.direction = direction
        elif direction == 'left' and self.direction != 'right':
            self.direction = direction
        elif direction == 'up' and self.direction != 'down':
            self.direction = direction
        elif direction == 'down' and self.direction != 'up':
            self.direction = direction
    def get_state(self):
        head = self.snake[0]
        food = self.food
        state = (head, food)
        return state

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        # Actualizar el valor Q del par estado-acción en la tabla 1
        if state not in q_table1:
            q_table1[state] = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
            q_table1[state][action] = q_table1.get(state, {}).get(action, 0) + alpha * (reward + gamma * max(q_table2.get(next_state, {}).values(), default=0) - q_table1.get(state, {}).get(action, 0))

        if state not in q_table2:
            q_table2[state] = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
            q_table2[state][action] = q_table2.get(state, {}).get(action, 0) + alpha * (reward + gamma * max(q_table1.get(next_state, {}).values(), default=0) - q_table2.get(state, {}).get(action, 0))


        # Actualizar el valor Q del par estado-acción en la tabla 2

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            # Exploración: seleccionar una acción aleatoria
            action = random.choice(['right', 'left', 'up', 'down'])
        else:
            # Explotación: seleccionar la mejor acción según las tablas Q
            if state in q_table1:
                action = max(q_table1[state], key=q_table1[state].get)
            else:
                action = random.choice(['right', 'left', 'up', 'down'])

        return action

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def play_game(self, alpha, gamma, epsilon, num_episodes, prev_distance):
        for episode in range(num_episodes):
            self.__init__()
            while True:
                print("q_table1", q_table1)
                print("q_table2", q_table2)
                state = self.get_state()

                action = self.choose_action(state, epsilon)
                self.set(action)
                self.move_snake()

                next_state = self.get_state()

                food_x, food_y = self.food

                distance = self.calculate_distance(self.snake[0], self.food)

                if self.check_collision():
                    reward = -1  # Penalizar colisiones
                    self.update_q_table(state, action, reward, next_state, alpha, gamma)
                    break
                elif len(self.snake) == WIDTH * HEIGHT / (SNAKE_SIZE ** 2):
                    reward = 1  # Recompensar si la serpiente llena todo el tablero
                    self.update_q_table(state, action, reward, next_state, alpha, gamma)
                    prev_distance = distance
                    break
                elif distance < prev_distance:
                    reward = 1  # Recompensar cuando la serpiente está cerca de la comida
                    self.update_q_table(state, action, reward, next_state, alpha, gamma)
                elif len(self.snake) > 1:
                    reward = 0.1  # Recompensar movimientos que no resulten en colisión ni en llenar todo el tablero

                    # Aplicar penalizaciones adicionales
                    head_x, head_y = self.snake[0]
                    dx = food_x - head_x
                    dy = food_y - head_y

                    if (
                            (action == 'left' and dx > 0) or
                            (action == 'right' and dx < 0) or
                            (action == 'up' and dy > 0) or
                            (action == 'down' and dy < 0)
                    ):
                        reward -= 0.5  # Penalización por moverse en la dirección opuesta a la comida

                    if (
                            (action == 'left' and head_x == 0) or
                            (action == 'right' and head_x == WIDTH - SNAKE_SIZE) or
                            (action == 'up' and head_y == 0) or
                            (action == 'down' and head_y == HEIGHT - SNAKE_SIZE)
                    ):
                        reward -= 0.2  # Penalización por moverse hacia los bordes de la pantalla

                    self.update_q_table(state, action, reward, next_state, alpha, gamma)

                else:
                    reward = 0
                    self.update_q_table(state, action, reward, next_state, alpha, gamma)

                prev_distance = distance
                performance.append(len(game.snake))
                convergence.append(max(q_table1.values(), key=max))

                screen.fill(BLACK)
                self.draw_snake()
                self.draw_food()
                self.update_score()
                pygame.display.update()
                clock.tick(SNAKE_SPEED)

# Parámetros de entrenamiento
alpha = 0.7  # Tasa de aprendizaje
gamma = 0.4  # Factor de descuento
epsilon = 0.3  # Tasa de exploración
num_episodes = 50


# Crear instancia del juego de la serpiente
game = SnakeGame()
prev_distance=99
# Entrenar el modelo utilizando Double Q-Learning
# Entrenar el modelo utilizando Double Q-Learning
game.play_game(alpha, gamma, epsilon, num_episodes, prev_distance)

# Graficar rendimiento
plt.figure()
plt.plot(range(len(performance)), performance)
plt.xlabel('Iteración')
plt.ylabel('Rendimiento')
plt.title('Rendimiento de la serpiente por iteración')
plt.show()

# Graficar convergencia
plt.figure()
plt.plot(range(num_episodes), convergence)
plt.xlabel('Episodio')
plt.ylabel('Valor máximo de Q')
plt.title('Convergencia de los valores Q por episodio')
plt.show()


