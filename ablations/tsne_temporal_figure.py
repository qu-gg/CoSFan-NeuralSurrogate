import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture

from sklearn.metrics import confusion_matrix
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np


task_ids = [i for i in range(9)] 

embeddings = dict()
labels = dict()

test_embeddings = dict()
test_labels = dict()

for task_id, scar_id in enumerate(task_ids):
    embeddings[task_id] = dict()
    labels[task_id] = dict()
    
    test_embeddings[task_id] = dict()
    test_labels[task_id] = dict()
    
    for task_id_next, scar_id_next in enumerate(task_ids):
        try:
            task_embedding = np.load(f"../experiments/feedforwardmask_synthetic_continual_er_5555_1.0/feedforwardmask/version_1/task_{task_id}/test_{task_id_next}_train/test_{task_id_next}_train_embeddings.npy", allow_pickle=True)
            print(task_id_next, task_embedding.shape)
            task_embedding = np.reshape(task_embedding, [task_embedding.shape[0], -1])
            embeddings[task_id][task_id_next] = task_embedding
            labels[task_id][task_id_next] = np.full([task_embedding.shape[0]], fill_value=task_id_next)
        except FileNotFoundError as e:
            print(e)
            continue
    
    for task_id_next, scar_id_next in enumerate(task_ids):
        try:
            task_embedding = np.load(f"../experiments/feedforwardmask_synthetic_continual_er_5555_1.0/feedforwardmask/version_1/task_{task_id}/test_{task_id_next}/test_{task_id_next}_embeddings.npy", allow_pickle=True)
            task_embedding = np.reshape(task_embedding, [task_embedding.shape[0], -1])
            test_embeddings[task_id][task_id_next] = task_embedding
            test_labels[task_id][task_id_next] = np.full([task_embedding.shape[0]], fill_value=task_id_next)
        except FileNotFoundError as e:
            print(e)
            continue
    

# Pad embeddings to same length
max_embed_length = 0
for key in embeddings.keys():
    for sub_key in embeddings[key].keys():
        if embeddings[key][sub_key].shape[1] > max_embed_length:
           max_embed_length = embeddings[key][sub_key].shape[1]

# Set random seed for reproducibility
np.random.seed(42)

for key in embeddings.keys():
    for sub_key in embeddings[key].keys():
        if embeddings[key][sub_key].shape[1] < max_embed_length:
            # Calculate padding width
            pad_width = max_embed_length - embeddings[key][sub_key].shape[1]
            
            # Generate random padding using the mean and std of the existing embeddings
            embed_mean = np.mean(embeddings[key][sub_key])
            embed_std = np.std(embeddings[key][sub_key])
            random_padding = np.random.normal(embed_mean, embed_std, 
                                           size=(embeddings[key][sub_key].shape[0], pad_width))
            
            # Concatenate the original embeddings with random padding
            embeddings[key][sub_key] = np.hstack([embeddings[key][sub_key], random_padding])
            test_embeddings[key][sub_key] = np.hstack([test_embeddings[key][sub_key], 
                                                     np.random.normal(embed_mean, embed_std,
                                                     size=(test_embeddings[key][sub_key].shape[0], pad_width))])


def refit_clusterer(clusterer, task_counter, combined_embeddings):
    # Fit Bayesian Gaussian Mixture model over the reservoir, taking their cluster assignments as pseudo-labels
    if clusterer is None:
        clusterer = BayesianGaussianMixture(n_components=1, max_iter=1000, n_init=3, random_state=1111)
        clusterer = clusterer.fit(combined_embeddings)
    else:        
        """ Refit with n+1 components with previous components """
        # Save the previous weights
        weights1 = clusterer.weights_
        means1 = clusterer.means_
        covariances1 = clusterer.covariances_

        # Rebuild the GMM, adding on another component
        clusterer = BayesianGaussianMixture(n_components=task_counter + 1, max_iter=1000, n_init=1,  random_state=1111)
        new_component_mean = np.mean(embeddings[task_counter][task_counter], axis=0)
        new_component_covariance = np.mean(covariances1, axis=0)  # Use the average of the existing covariances

        clusterer.weights_ = np.concatenate([weights1 * (1 - 1e-1), [1e-1]])
        clusterer.means_ = np.vstack([means1, new_component_mean])
        clusterer.covariances_ = np.concatenate([covariances1, [new_component_covariance]])
        clusterer.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(clusterer.covariances_))

        # Do the fit
        clusterer = clusterer.fit(combined_embeddings)
        
        # """ Detect inactive components, throw them out """
        # # Step 2: Detect inactive components (weights close to zero)
        # threshold = 0.05
        # active_indices = np.where(clusterer.weights_ > threshold)[0]

        # # Step 3: Extract parameters of active components
        # active_weights = clusterer.weights_[active_indices]
        # active_means = clusterer.means_[active_indices]
        # active_covariances = clusterer.covariances_[active_indices]

        # # Normalize the weights of the active components
        # active_weights /= active_weights.sum()

        # # Step 4: Initialize a new BayesianGaussianMixture with the number of active components
        # clusterer = BayesianGaussianMixture(
        #     n_components=len(active_indices), max_iter=1000, n_init=1,
        #     weight_concentration_prior=0.01, random_state=1111
        # )

        # clusterer.weights_ = active_weights
        # clusterer.means_ = active_means
        # clusterer.covariances_ = active_covariances
        # clusterer.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(active_covariances))
        # clusterer = clusterer.fit(combined_embeddings)

    return clusterer


# Initialize clusterer
clusterer = None
selected_tasks = [0, 1, 4, 8]  

# First, fit GMM continually on all tasks
for task_id in range(max(selected_tasks) + 1):
    # Get embeddings for current task
    known_embed_stack = np.vstack([embeddings[task_id][sub_key] 
                                 for sub_key in range(len(task_ids[:task_id + 1]))])
    
    # Update GMM
    clusterer = refit_clusterer(clusterer, task_id, known_embed_stack)
    
    # Store clusterer state if this is a task we want to plot
    if task_id in selected_tasks:
        if 'clusterer_states' not in locals():
            clusterer_states = {}
        clusterer_states[task_id] = {
            'means': clusterer.means_.copy(),
            'covariances': clusterer.covariances_.copy(),
            'weights': clusterer.weights_.copy()
        }

# Third section: Create visualizations
fig = plt.figure(figsize=(20, 15))

for plot_idx, task_counter in enumerate(selected_tasks):
    # Get known and unknown stacks for this task (original high-dim data)
    known_embed_stack = np.vstack([embeddings[task_counter][sub_key] 
                                 for sub_key in range(len(task_ids[:task_counter + 1]))])
    known_label_stack = np.concatenate([labels[task_counter][sub_key] 
                                      for sub_key in range(len(task_ids[:task_counter + 1]))])

    test_known_embed_stack = np.vstack([test_embeddings[task_counter][sub_key] 
                                      for sub_key in range(len(task_ids[:task_counter + 1]))])
    test_known_label_stack = np.concatenate([test_labels[task_counter][sub_key] 
                                           for sub_key in range(len(task_ids[:task_counter + 1]))])

    # Handle unknown stacks (original high-dim data)
    if task_counter < len(task_ids) - 1:
        unknown_embed_stack = np.vstack([embeddings[task_counter][sub_key] 
                                       for sub_key in range(task_counter + 1, len(task_ids))])
        unknown_label_stack = np.concatenate([labels[task_counter][sub_key] 
                                            for sub_key in range(task_counter + 1, len(task_ids))])
    else:
        unknown_embed_stack = np.empty((0, known_embed_stack.shape[1]))
        unknown_label_stack = np.array([])

    # Get clusterer state for this task
    clusterer_state = clusterer_states[task_counter]
    
    # Combine all data for TSNE (all in original high-dim space)
    if unknown_embed_stack.size > 0:
        all_data = np.vstack((known_embed_stack, unknown_embed_stack, 
                             test_known_embed_stack, clusterer_state['means']))
    else:
        all_data = np.vstack((known_embed_stack, test_known_embed_stack, 
                             clusterer_state['means']))
    
    # Single TSNE transform for all data
    tsne = TSNE(n_components=2, perplexity=50, metric="cosine", random_state=3, verbose=False)
    tsne_embedding = tsne.fit_transform(all_data)
    
    # Split embeddings back into respective components
    n_centers = clusterer_state['means'].shape[0]  # Use stored state
    if unknown_embed_stack.size > 0:
        split_idx1 = known_embed_stack.shape[0]
        split_idx2 = split_idx1 + unknown_embed_stack.shape[0]
        split_idx3 = split_idx2 + test_known_embed_stack.shape[0]
        
        known_embed_stack_2d = tsne_embedding[:split_idx1]
        unknown_embed_stack_2d = tsne_embedding[split_idx1:split_idx2]
        test_known_embed_stack_2d = tsne_embedding[split_idx2:split_idx3]
        cluster_centers_2d = tsne_embedding[split_idx3:]
    else:
        split_idx1 = known_embed_stack.shape[0]
        split_idx2 = split_idx1 + test_known_embed_stack.shape[0]
        
        known_embed_stack_2d = tsne_embedding[:split_idx1]
        test_known_embed_stack_2d = tsne_embedding[split_idx1:split_idx2]
        cluster_centers_2d = tsne_embedding[split_idx2:]
        unknown_embed_stack_2d = np.empty((0, 2))

    # Plot in 2D
    ax = plt.subplot(2, 2, plot_idx + 1)
    
    scatter1 = ax.scatter(known_embed_stack_2d[:, 0], known_embed_stack_2d[:, 1], 
                         marker='x', c=known_label_stack, label='Train')
    scatter2 = ax.scatter(test_known_embed_stack_2d[:, 0], test_known_embed_stack_2d[:, 1], 
                         marker='x', c=test_known_label_stack, label='Test')
    if unknown_embed_stack.size > 0:
        scatter3 = ax.scatter(unknown_embed_stack_2d[:, 0], unknown_embed_stack_2d[:, 1], 
                            marker='o', c=unknown_label_stack, label='Unknown')
    
    # Plot cluster centers
    scatter4 = ax.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                         c='black', marker='D', s=100, label='Cluster Center')


    # Customize subplot
    ax.set_title(f"Task {task_counter}", weight='bold', fontsize=15)
    # if plot_idx in [0, 2]:  # Left column
    #     ax.set_ylabel("tSNE-2", weight='bold', fontsize=12)
    # if plot_idx in [2, 3]:  # Bottom row
    #     ax.set_xlabel("tSNE-1", weight='bold', fontsize=12)


# Update legend handling
handles1, labels1 = scatter1.legend_elements()
handles2, labels2 = scatter2.legend_elements()
handles4, labels4 = scatter4.legend_elements()
handles = handles1 + handles2 + handles4
labels = labels1 + labels2 + ['Cluster Centers']

if 'scatter3' in locals():
    handles3, labels3 = scatter3.legend_elements()
    handles = handles + handles3
    labels = labels + labels3

# # Add single legend for entire figure
# fig.legend(handles, labels, 
#           loc='center', 
#           bbox_to_anchor=(0.5, 0.05),
#           ncol=3, 
#           fontsize=12,
#           frameon=True,
#           fancybox=True,
#           shadow=True)


# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend

# Save figure
plt.savefig("tsne_temporal_combined.png", dpi=300, bbox_inches='tight')
plt.close()
