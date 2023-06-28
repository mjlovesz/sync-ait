class SeqHandler:
    @staticmethod
    def sort_api_sequences(seqs):
        idx = 0
        cnt = len(seqs)
        i = 0
        visited_nodes = []
        while i < cnt:
            idx_flag = True
            chk_flag = False

            item = seqs[idx]
            if item in visited_nodes:
                idx += 1
                continue
            visited_nodes.append(item)

            item_idx = len(item[1])
            for j in range(idx + 1, cnt):
                cur_idx = len(seqs[j][1])
                if item_idx < cur_idx:
                    if j + 1 == cnt:
                        seqs.pop(idx)
                        seqs.insert(j, item)
                        idx_flag = False
                        break
                    else:
                        chk_flag = True
                        continue
                else:
                    if chk_flag:
                        seqs.pop(idx)
                        seqs.insert(j - 1, item)
                        idx_flag = False
                        break
                    else:
                        if item_idx == cur_idx:
                            break
            if idx_flag:
                idx += 1
            i += 1

    @staticmethod
    def filter_api_sequences(seqs):
        for seq in seqs:
            pass

    @staticmethod
    def format_api_sequences(seqs):
        pass

    @staticmethod
    def union_api_sequences(seqs):
        pass
