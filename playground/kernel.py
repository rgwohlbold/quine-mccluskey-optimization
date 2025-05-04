#minterms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 18, 19, 62, 63]
minterms = list(range(0, 64))

bitmask = 0
for minterm in minterms:
    bitmask |= 1 << minterm


impl1 = bitmask
impl2 = bitmask >> 1
merged1 = (impl1 & impl2) & 0b01010101010101010101010101010101010101010101010101010101010101010101010101010101
merged1_ = merged1 >> 1
merged2 = (merged1 | merged1_) & 0b0011001100110011001100110011001100110011001100110011001100110011
merged2_ = merged2 >> 2
merged3 = (merged2 | merged2_) & 0x0F0F0F0F0F0F0F0F
merged3_ = merged3 >> 4
merged4 = (merged3 | merged3_) & 0x00FF00FF00FF00FF
merged4_ = merged4 >> 8
merged5 = (merged4 | merged4_) & 0x0000FFFF0000FFFF
merged5_ = merged5 >> 16
merged6 = (merged5 | merged5_) & 0x00000000FFFFFFFF


print("merged1", bin(merged1))
print("merged2 ", bin(merged2))
print("merged3   ", bin(merged3))
print("merged4       ", bin(merged4))
print("merged5               ", bin(merged5))
print("merged6                               ", bin(merged6))