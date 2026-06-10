import torch


def main():
    # load the library, assume it is located together with this file
    torch.ops.load_library("../libCANDOR.so")
    # gen the input tensor
    torch.ops.CANDOR.index_create("idx1", "flat")
    torch.ops.CANDOR.index_ediCfgI64("idx1", "vecDim", 4)
    a = torch.rand(1, 4)
    b = torch.rand(1, 4)
    torch.ops.CANDOR.index_init("idx1")
    torch.ops.CANDOR.index_insert("idx1", a)
    torch.ops.CANDOR.index_insert("idx1", b)
    c = torch.ops.CANDOR.index_search("idx1", a, 1)
    print("rawData", torch.ops.CANDOR.index_rawData("idx1"))
    print("search result", c)
    torch.ops.CANDOR.tensorToFile(c[0], "c.rbt")
    print("loaded result")
    d = torch.ops.CANDOR.tensorFromFile("c.rbt")
    print(d)


if __name__ == "__main__":
    main()
