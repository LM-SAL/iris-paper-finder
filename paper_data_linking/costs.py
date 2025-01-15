import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tiktoken
from tqdm import tqdm

from paper_data_linking.enums import Locations

rng = np.random.default_rng()

enc = tiktoken.encoding_for_model("gpt-4")
token_lengths = []
for infile in tqdm(Locations.TEXTS.value.iterdir()):
    with open(infile, "r") as f0:
        txt = f0.read()
        token_lengths.append(len(enc.encode(txt)))

# Number of bootstrapped samples to create
num_bootstraps = 1000  # You can adjust this value based on your needs
bootstrapped_sums = np.zeros(num_bootstraps)

num_total_papers = 6500
# Perform bootstrapping
for i in range(num_bootstraps):
    # Generate a bootstrap sample by sampling with replacement from the data points
    bootstrap_sample = np.random.choice(
        token_lengths, size=num_total_papers, replace=True)
    bootstrap_sum = np.sum(bootstrap_sample)
    bootstrapped_sums[i] = bootstrap_sum

mean_sum = np.mean(bootstrapped_sums)
# estimated_total_sum = mean_sum * (7000 / len(token_lengths))

# plt.clf()
sns.set_style("whitegrid")  # Setting the grid style
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.distplot(bootstrapped_sums)
plt.xlabel('total tokens')
plt.ylabel('frequency')
plt.show()

# Price per token
gpt4_input_price = 0.03 / 1e3
gpt4_output_price = 0.06 / 1e3
ada_price = 0.0004 / 1e3

output_frac_input = 0.05  # assume output is approx 5% of input tokens
cost_input_samples = bootstrapped_sums * gpt4_input_price
cost_output_samples = bootstrapped_sums * gpt4_output_price
cost_samples = cost_input_samples + cost_output_samples

fraction_query_tokens = 0.1

embed_price = bootstrapped_sums * ada_price
query_price = bootstrapped_sums * gpt4_input_price * fraction_query_tokens
response_price = bootstrapped_sums * gpt4_output_price * output_frac_input
cost_samples_embeddings = embed_price + query_price + response_price

# Create the plot
# plt.clf()
sns.set_style("whitegrid")  # Setting the grid style
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

# Plotting the histogram
sns.distplot(cost_samples, kde=True, hist=True, rug=False)

# Adding a title
plt.title("Distribution of Cost in Dollars")

# Estimating Gaussian mean and variance
mean = np.mean(cost_samples)
variance = np.var(cost_samples)

# Annotating the plot
plt.annotate(f"Mean: {mean:.2f}", xy=(0.65, 0.85), xycoords="axes fraction")
plt.annotate(f"Variance: {variance:.2f}", xy=(0.65, 0.8), xycoords="axes fraction")

plt.show()

# Create the plot
# plt.clf()
sns.set_style("whitegrid")  # Setting the grid style
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

# Plotting the histogram
sns.distplot(cost_samples_embeddings, kde=True, hist=True, rug=False)

# Adding a title
plt.title("Distribution of Cost in Dollars using Embeddings")

# Estimating Gaussian mean and variance
mean = np.mean(cost_samples_embeddings)
variance = np.var(cost_samples_embeddings)

# Annotating the plot
plt.annotate(f"Mean: {mean:.2f}", xy=(0.65, 0.85), xycoords="axes fraction")
plt.annotate(f"Variance: {variance:.2f}", xy=(0.65, 0.8), xycoords="axes fraction")

plt.show()

print("Done part 1")


# # Parameters
# num_papers = 1000  # Fixed number of papers
# mu_chunks_per_paper, sigma_chunks_per_paper = 10, 2  # Mean and standard deviation for C
# mu_tokens_per_chunk, sigma_tokens_per_chunk = 500, 100  # Mean and standard deviation for T
# chunks_per_query = np.array([0.25, 0.5, 0.25])  # Discrete distribution for M
# price_per_token = 0.06 / 1e3  # Price per token
#
# # Generate samples from the truncated normal distributions
# num_samples = 1e5
# C_samples = np.round(truncnorm.rvs(a=0, b=np.inf, loc=mu_chunks_per_paper, scale=sigma_chunks_per_paper, size=num_samples))
# T_samples = np.round(truncnorm.rvs(a=0, b=np.inf, loc=mu_tokens_per_chunk, scale=sigma_tokens_per_chunk, size=num_samples))
#
# # Compute the CT_samples
# CT_samples = C_samples * T_samples
#
# # Compute the mixture of CTM_samples
# M_values = np.arange(1, len(chunks_per_query) + 1)
# CTM_samples = np.dot(CT_samples[:, np.newaxis], M_values[np.newaxis, :]) * chunks_per_query
#
# # Calculate the expected total cost
# expected_total_cost = num_papers * np.sum(CTM_samples) * price_per_token / num_samples
# print("Expected total cost:", expected_total_cost)
#
# # Plot the histogram of N * CTM_samples (scaled by price_per_token)
# plt.hist(num_papers * CTM_samples.ravel() * price_per_token, bins=100, density=True, alpha=0.6, color='blue')
# plt.xlabel('Total Cost')
# plt.ylabel('Probability Density')
# plt.title('Expected Costs Probability Distribution')
# plt.show()


# num_papers = 7000
# chunks_mean, chunks_var = 10, 15
# tokens_mean, tokens_var = 250, 10_000
#
# # plot_pmf(chunks_params, max(num_chunks), 'PMF of Number of Chunks')
# # plot_pmf(tokens_params, max(max(lengths) for lengths in token_lengths), 'PMF of Token Lengths')
#
# # Calculate parameters for negative binomial distribution
# chunks_params = calculate_params(chunks_mean, chunks_var)
# tokens_params = calculate_params(tokens_mean, tokens_var)
#
# n = int(1e3)
# num_tokens_samples = np.zeros(n)
# for i in tqdm(range(n)):
#     num_tokens_samples[i] = sample_num_tokens(num_papers, chunks_params, tokens_params)
# Function to calculate parameters of negative binomial distribution
# def calculate_params(mean, var):
#     p = mean / (mean + var)
#     r = mean**2 / (var - mean)
#     return p, r
#
#
# # Function to generate data for each paper
# def generate_data(num_papers, chunks_params, tokens_params):
#     num_chunks = []
#     token_lengths = []
#     for _ in range(num_papers):
#         num_chunks_paper = np.random.negative_binomial(chunks_params[1], chunks_params[0])
#         num_chunks.append(num_chunks_paper)
#         token_lengths_paper = np.random.negative_binomial(tokens_params[1], tokens_params[0], size=num_chunks_paper)
#         token_lengths.append(token_lengths_paper)
#     return num_chunks, token_lengths
#
#
# # Function to plot PMF
# def plot_pmf(params, max_val, title):
#     n = np.arange(0, max_val)
#     pmf = np.random.negative_binomial(params[1], params[0], size=max_val)
#     plt.bar(n, pmf)
#     plt.title(title)
#     plt.show()
#
#
# def sample_num_tokens(num_papers, chunks_params, token_params):
#     num_chunks, token_lengths = generate_data(num_papers, chunks_params, tokens_params)
#     num_tokens = sum([cl.sum() for cl in token_lengths])
#
#     return num_tokens
#
