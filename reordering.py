reor_file = open('./verbmobil_reordering_withPunc/dev_reordering')
src_file =  open('./verbmobil_reordering_withPunc/Dev.fa','r')
src_lines = src_file.readlines()
new_src_file = open('./verbmobil_reordering_withPunc/Dev.reorder','w')



i = 0
for reor_line in reor_file:
    print(i+1)
    src_dict = set([])
    tuples = reor_line.split()
    src_line = src_lines[i]
    src_vocabs = src_line.split()
    new_line = []
    target_indexes = []

    for tuple in tuples:
        indexes = tuple.split('-')
        src_index = int(indexes[0])
        if src_index not in src_dict:
            src_dict.add(src_index)
            

            new_line.append(src_vocabs[src_index])
    if len(src_dict) < len(src_vocabs):
        j=0
        for vocab in src_vocabs:
            if not j in src_dict:
                
                src_dict.add(j)
                new_line.insert(j,vocab)
            j+=1
    new_line.append('\n')
    new_src_file.write(' '.join([str(x) for x in new_line]))
    i+=1






