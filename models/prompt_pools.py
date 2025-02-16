
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptPool(nn.Module):
    def __init__(self, num_prompts=20, prompt_dim=1024, model_input_embeds=None, p=0.1):
        super().__init__()
        self.num_prompts = num_prompts
        
        # Initialize keys with small random values
        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.01)  # Learnable keys
        
        # Initialize values based on model input embeddings mean plus noise
        if model_input_embeds is not None:
            self.prompt_values = nn.Parameter(model_input_embeds)
        else:
            self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))
            
        self.dropout = None
        if p > 0:
            self.dropout = torch.nn.Dropout(p)
            
    def compute_cosine_similarity(self, input_embedding):
        """Compute cosine similarities between normalized input_embedding and prompt keys."""
        norm_input = F.normalize(input_embedding, dim=-1)       # [B, prompt_dim]
        norm_keys = F.normalize(self.prompt_keys, dim=-1)         # [num_prompts, prompt_dim]
        return torch.matmul(norm_input, norm_keys.T)              # [B, num_prompts]

    def forward(self, input_embedding, top_k=5):
        """
        Selects the top-k relevant prompts based on similarity with the input.
        Arguments:
        - input_embedding: [batch_size, hidden_dim]
        - top_k: Number of prompts to select
        """
        # Compute similarities between input and prompt keys
        similarities = self.compute_cosine_similarity(input_embedding)  # [batch_size, num_prompts]
            
        normalized_similarities = F.softmax(similarities, dim=1)
        if self.dropout is not None:
            normalized_similarities = self.dropout(normalized_similarities)
            normalized_similarities = F.softmax(similarities, dim=1)
            
        # Select top-k indices and corresponding values
        topk_values, topk_indices = torch.topk(normalized_similarities, top_k, dim=1)
        
        # Gather the selected prompt values
        selected_prompts = self.prompt_values[topk_indices]  # [B, top_k, prompt_dim]
        
        # Compute diversity loss as the sum of the top-k similarity values, averaged over the batch
        diversity_loss = - topk_values.sum(dim=1).mean()
        
        selected_prompts = self.prompt_values[topk_indices]  # [batch_size, top_k, prompt_dim]
            
        return selected_prompts, diversity_loss, topk_indices
    

class StaticPromptPool(nn.Module):
    def __init__(self, num_prompts=20, prompt_dim=1024, model_input_embeds=None, p=0.1,
                tasks=["QA", "asr", "emotion_recognition"]):
        super().__init__()
        self.num_prompts = num_prompts
        
        # Initialize keys with small random values
        self.prompt_keys = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.01)  # Learnable keys
        
        # Initialize values based on model input embeddings mean plus noise
        if model_input_embeds is not None:
            self.prompt_values = nn.Parameter(model_input_embeds)
        else:
            self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))
            
        self.dropout = None
        if p > 0:
            self.dropout = torch.nn.Dropout(p)
            
        self.num_tasks = len(tasks)
        self.tasks = tasks
        
        self.start_indices = {}
        for i, task in enumerate(self.tasks):
            self.start_indices[task] = i * int(0.6 * self.num_prompts / (len(self.tasks) - 1))
            
    def forward(self, input_embedding, top_k=5, tasks = []):
        """
        Selects the top-k relevant prompts based on similarity with the input.
        Arguments:
        - input_embedding: [batch_size, hidden_dim]
        - top_k: Number of prompts to select
        """
        start_indices = torch.tensor([self.start_indices[task] for task in tasks], device=self.prompt_values.device)
        diversity_loss = 0.0
        
        # Generate indices for each task's token range
        task_token_range = torch.arange(top_k, device=self.prompt_values.device)  # [top_k]
        topk_indices = start_indices.unsqueeze(1) + task_token_range.unsqueeze(0)  # [batch_size, top_k]

        # Gather the corresponding prompt values
        selected_prompts = self.prompt_values[topk_indices]  # [batch_size, top_k, prompt_dim]
            
        return selected_prompts, diversity_loss, topk_indices
    
    

class StaticPromptPool2(nn.Module):
    def __init__(self, num_prompts=20, prompt_dim=1024, model_input_embeds=None, p=0.1,
                 tasks=["QA", "asr", "emotion_recognition"], f=0.4):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.f = f  # Fraction of pool for max prompt selection
        
        # Compute sizes for disjoint and shared portions
        self.x = int(((1 - f) / (self.num_tasks - 1)) * self.num_prompts)  # Disjoint portion per task
        self.y = int((self.num_prompts * (f * self.num_tasks - 1)) / (self.num_tasks - 1))  # Shared portion
        
        assert self.x * (self.num_tasks - 1) + self.y <= self.num_prompts, "Invalid split of prompt pool!"

        # Initialize values based on model input embeddings mean plus noise
        if model_input_embeds is not None:
            self.prompt_values = nn.Parameter(model_input_embeds)
        else:
            self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))

        # Assign starting indices for each task
        self.start_indices = {}
        for i, task in enumerate(self.tasks):
            self.start_indices[task] = i * self.x  # Each task gets a disjoint segment

        # Shared portion starts after disjoint parts
        self.shared_start_idx = self.x * self.num_tasks

    def forward(self, input_embedding, top_k=5, tasks=[]):
        """
        Selects relevant prompts for the input based on task mapping.
        Arguments:
        - input_embedding: [batch_size, hidden_dim] (not used in this version)
        - top_k: Number of prompts to select (<= f * s)
        - tasks: List of task names corresponding to batch instances
        """
        batch_size = len(tasks)
        start_indices = torch.tensor([self.start_indices[task] for task in tasks], device=self.prompt_values.device)

        # Compute how many tokens come from the disjoint portion and how many from the shared portion
        disjoint_k = min(self.x, top_k)  # Max we can take from disjoint portion
        shared_k = max(0, top_k - disjoint_k)  # Remaining tokens come from shared portion

        # Get disjoint portion indices
        task_token_range = torch.arange(disjoint_k, device=self.prompt_values.device)  # [disjoint_k]
        disjoint_indices = start_indices.unsqueeze(1) + task_token_range.unsqueeze(0)  # [batch_size, disjoint_k]

        # Get shared portion indices if needed
        if shared_k > 0:
            shared_range = torch.arange(shared_k, device=self.prompt_values.device)  # [shared_k]
            shared_indices = self.shared_start_idx + shared_range.unsqueeze(0)  # [1, shared_k] (broadcasts over batch)
            shared_indices = shared_indices.expand(batch_size, -1)  # [batch_size, shared_k]

            # Concatenate disjoint and shared portions
            topk_indices = torch.cat([disjoint_indices, shared_indices], dim=1)  # [batch_size, top_k]
        else:
            topk_indices = disjoint_indices  # Only disjoint portion is used

        # Gather the corresponding prompt values
        selected_prompts = self.prompt_values[topk_indices]  # [batch_size, top_k, prompt_dim]

        diversity_loss = 0.0  # No diversity loss in this version

        return selected_prompts, diversity_loss, topk_indices
    
    
    

class StaticPromptPool3(nn.Module):
    def __init__(self, num_prompts=20, prompt_dim=1024, model_input_embeds=None, p=0.1,
                 tasks=["QA", "asr", "emotion_recognition"], f=0.4):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.f = f  # Fraction of pool for max prompt selection
        
        # Compute sizes for disjoint and shared portions
        self.x = int(((1 - f) / (self.num_tasks - 1)) * self.num_prompts)  # Disjoint portion per task
        self.y = int((self.num_prompts * (f * self.num_tasks - 1)) / (self.num_tasks - 1))  # Shared portion
        
        assert self.x * (self.num_tasks - 1) + self.y <= self.num_prompts, "Invalid split of prompt pool!"

        # Initialize values based on model input embeddings mean plus noise
        if model_input_embeds is not None:
            self.prompt_values = nn.Parameter(model_input_embeds)
        else:
            self.prompt_values = nn.Parameter(torch.randn(num_prompts, prompt_dim))

        # Assign starting indices for each task
        self.start_indices = {}
        for i, task in enumerate(self.tasks):
            self.start_indices[task] = i * self.x  # Each task gets a disjoint segment

        # Shared portion starts after disjoint parts
        self.shared_start_idx = self.x * self.num_tasks
        
    def forward(self, input_embedding, top_k=5, tasks=[]):
        """
        Selects relevant prompts for the input based on task mapping.
        Arguments:
        - input_embedding: [batch_size, hidden_dim] (not used in this version)
        - top_k: Number of prompts to select (<= f * s)
        - tasks: List of task names corresponding to batch instances
        """
        batch_size = len(tasks)
        start_indices = torch.tensor([self.start_indices[task] for task in tasks], device=self.prompt_values.device)

        # Compute how many tokens come from the shared portion and how many from the disjoint portion
        shared_k = min(self.y, top_k)  # Max we can take from shared portion
        disjoint_k = max(0, top_k - shared_k)  # Remaining tokens come from disjoint portion

        # Get shared portion indices
        shared_range = torch.arange(shared_k, device=self.prompt_values.device)  # [shared_k]
        shared_indices = self.shared_start_idx + shared_range.unsqueeze(0)  # [1, shared_k] (broadcasts over batch)
        shared_indices = shared_indices.expand(batch_size, -1)  # [batch_size, shared_k]

        # Get disjoint portion indices if needed
        if disjoint_k > 0:
            task_token_range = torch.arange(disjoint_k, device=self.prompt_values.device)  # [disjoint_k]
            disjoint_indices = start_indices.unsqueeze(1) + task_token_range.unsqueeze(0)  # [batch_size, disjoint_k]

            # Concatenate shared and disjoint portions
            topk_indices = torch.cat([shared_indices, disjoint_indices], dim=1)  # [batch_size, top_k]
        else:
            topk_indices = shared_indices  # Only shared portion is used

        # Gather the corresponding prompt values
        selected_prompts = self.prompt_values[topk_indices]  # [batch_size, top_k, prompt_dim]

        diversity_loss = 0.0  # No diversity loss in this version

        return selected_prompts, diversity_loss, topk_indices
