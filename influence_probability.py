import torch

H = torch.load("node_embeddings.pt")


def influence_probability(u, v):
    return torch.sigmoid((H[v] * H[u]).sum(dim=-1))


def main():
    u = 0
    v = 1
    print(influence_probability(u, v))


if __name__ == "__main__":
    main()
