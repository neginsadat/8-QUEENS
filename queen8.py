import numpy as np
import matplotlib.pyplot as plt

# تابع برازندگی: شمارش تعداد برخوردهای وزیرها
def fitness(chromosome):
    attacks = 0
    n = len(chromosome)
    for i in range(n):
        for j in range(i + 1, n):
            # اگر در یک قطر باشند، برخورد دارند
            if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                attacks += 1
    return attacks

# ایجاد جمعیت اولیه به صورت تصادفی
def init_population(pop_size, n):
    return np.array([np.random.permutation(n) for _ in range(pop_size)])

# انتخاب والدین: بهترین ۲ از ۵ فرد تصادفی
def select_parents(population, fitness_values):
    indices = np.random.choice(len(population), 5, replace=False)
    selected = population[indices]
    selected_fitness = fitness_values[indices]
    best_two_indices = np.argsort(selected_fitness)[:2]
    return selected[best_two_indices]

# تابع کراس اور (برش و پر کردن)
def crossover(parent1, parent2):
    n = len(parent1)
    cut = np.random.randint(1, n-1)
    child1 = np.concatenate((parent1[:cut], [x for x in parent2 if x not in parent1[:cut]]))
    child2 = np.concatenate((parent2[:cut], [x for x in parent1 if x not in parent2[:cut]]))
    return child1, child2

# تابع جهش: جابجایی تصادفی دو ژن
def mutate(chromosome, mutation_prob=0.2):
    if np.random.random() < mutation_prob:
        i, j = np.random.choice(len(chromosome), 2, replace=False)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

# تابع انتخاب بقا با استفاده از الیتیسم
def elitism_selection(population, offspring, fitness_values, elite_size=2):
    # مرتب‌سازی جمعیت و فرزندان بر اساس برازندگی
    sorted_indices = np.argsort(fitness_values)
    best_individuals = population[sorted_indices][:elite_size]
    new_population = np.concatenate([best_individuals, offspring[elite_size:]])
    return new_population

# الگوریتم ژنتیک
def genetic_algorithm(n=8, pop_size=100, max_evals=10000, mutation_prob=0.2, elitism=False):
    population = init_population(pop_size, n)
    fitness_values = np.array([fitness(ind) for ind in population])
    
    evaluations = 0
    fitness_history = []
    max_fitness_history = []
    
    while evaluations < max_evals:
        new_population = []
        
        for _ in range(pop_size // 2):
            parents = select_parents(population, fitness_values)
            child1, child2 = crossover(parents[0], parents[1])
            mutate(child1, mutation_prob)
            mutate(child2, mutation_prob)
            new_population.extend([child1, child2])
        
        new_population = np.array(new_population)
        new_fitness_values = np.array([fitness(ind) for ind in new_population])
        
        if elitism:
            # استفاده از الیتیسم برای جایگزینی جمعیت
            population = elitism_selection(population, new_population, fitness_values, elite_size=2)
            fitness_values = np.array([fitness(ind) for ind in population])
        else:
            # جایگزینی نسل به صورت عادی
            population = new_population
            fitness_values = new_fitness_values
        
        best_fitness = np.min(fitness_values)
        avg_fitness = np.mean(fitness_values)
        
        fitness_history.append(avg_fitness)
        max_fitness_history.append(best_fitness)
        
        if best_fitness == 0:
            print(" we found the solution!")
            print(population[np.argmin(fitness_values)])
            break
        
        evaluations += pop_size
    
    return fitness_history, max_fitness_history
# اجرای الگوریتم ژنتیک و ثبت برازندگی‌ها
fitness_history, max_fitness_history = genetic_algorithm(n=8, pop_size=100, max_evals=10000)

# رسم نمودار برازندگی میانگین و حداکثر در هر نسل
plt.plot(fitness_history, label="برازندگی میانگین")
plt.plot(max_fitness_history, label="برازندگی حداکثر (بهترین)")
plt.xlabel("نسل")
plt.ylabel("برازندگی")
plt.title("روند تغییر برازندگی در هر نسل")
plt.legend()
plt.show()

# اجرای الگوریتم و نمایش نمودارها برای تاثیر جهش
fitness_history_20, max_fitness_history_20 = genetic_algorithm(n=8, pop_size=100, max_evals=10000, mutation_prob=0.2)
fitness_history_50, max_fitness_history_50 = genetic_algorithm(n=8, pop_size=100, max_evals=10000, mutation_prob=0.5)
fitness_history_100, max_fitness_history_100 = genetic_algorithm(n=8, pop_size=100, max_evals=10000, mutation_prob=1.0)

# رسم نمودار برازندگی میانگین برای سه حالت جهش
plt.plot(fitness_history_20, label="mutation 20%")
plt.plot(fitness_history_50, label="mutation 50%")
plt.plot(fitness_history_100, label="mutation 100%")
plt.xlabel("generation")
plt.ylabel(" mean fitness")
plt.title("mutation vs fitness")
plt.legend()
plt.show()


# اجرای الگوریتم با روش نسل‌بندی
fitness_history_no_elitism, max_fitness_history_no_elitism = genetic_algorithm(n=8, pop_size=100, max_evals=10000)

# اجرای الگوریتم با الیتیسم
fitness_history_elitism, max_fitness_history_elitism = genetic_algorithm(n=8, pop_size=100, max_evals=10000, elitism=True)

# رسم نمودار مقایسه
plt.plot(fitness_history_no_elitism, label="Generational Replacement - Average Fitness")
plt.plot(max_fitness_history_no_elitism, label="Generational Replacement - Best Fitness")

plt.plot(fitness_history_elitism, label="Elitism - Average Fitness")
plt.plot(max_fitness_history_elitism, label="Elitism - Best Fitness")

plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Comparison of Generational Replacement vs. Elitism")
plt.legend()
plt.show()

# اجرای الگوریتم برای N=10
fitness_history_n10, max_fitness_history_n10 = genetic_algorithm(n=10, pop_size=100, max_evals=10000)

# اجرای الگوریتم برای N=12
fitness_history_n12, max_fitness_history_n12 = genetic_algorithm(n=12, pop_size=100, max_evals=10000)

# اجرای الگوریتم برای N=20
fitness_history_n20, max_fitness_history_n20 = genetic_algorithm(n=20, pop_size=100, max_evals=10000)

# رسم نمودار برای N=10، N=12 و N=20
plt.plot(fitness_history_n10, label="N=10 - Average Fitness")
plt.plot(max_fitness_history_n10, label="N=10 - Best Fitness")

plt.plot(fitness_history_n12, label="N=12 - Average Fitness")
plt.plot(max_fitness_history_n12, label="N=12 - Best Fitness")

plt.plot(fitness_history_n20, label="N=20 - Average Fitness")
plt.plot(max_fitness_history_n20, label="N=20 - Best Fitness")

plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Scalability Test: Effect of N on Fitness")
plt.legend()
plt.show()

