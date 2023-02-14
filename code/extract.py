# Function: Given the raw data file, you will get the target file, which contains the trunk of each sentence.
# Author: fuxueli mingzhishao chunchengchi
# Email:lifuxue119@163.com
# Completion time: September 10, 2022
import sys
from nltk.tree import *
from stanfordcorenlp import StanfordCoreNLP
target_tree_dict={'NP':0} # 'target_tree_dict' stores the root node of the desired subtree
def judge(localtree):
    """Judge whether subtree is a single chain.
    Args:
        localtree: str. The input tree.
    return:
        If the tree is single chain return True,else return False.
    """
    flag=0
    new_flag=1
    for subtree_i in Tree.fromstring(localtree).subtrees():
        flag+=1
    while type(Tree.fromstring(localtree)[0])!=str:
        new_flag+=1
        localtree=str(Tree.fromstring(localtree)[0])
    if flag==new_flag:
        return True
    else:
        return False

def judgeNode(subtree,out_result):
    """Extract the required leaf nodes from the subtree and add them to sentence trunk.
    Args:
        subtree: str. The input tree.
        out_result: str. Intermediate result of sentence trunk.
    return:
        out_result: str. New template that add extracting words from the subtree.
    """
    pending_subtree=Tree.fromstring(subtree)
    # If it is the desired subtree, add the required leaf node in the subtree to the sentence trunk.
    if (pending_subtree.label() in target_tree_dict) and len(pending_subtree.leaves())<=6:
        noun=extract_noun(str(pending_subtree))
        if noun==None:
            noun=further_extract_noun(str(pending_subtree))
            if noun==None:
                noun=pending_subtree.leaves()[0]
        out_result=out_result+noun+' '
    # Otherwise, traverse its subtree.
    else:
        for subtree_i in pending_subtree:
            subtree_i=Tree.fromstring(str(subtree_i))
            # If it is a single chain, add the leaf node directly to the sentence trunk.
            if judge(str(subtree_i)):
                subtree_i=subtree_i.leaves()
                out_result=out_result+str(subtree_i[0])+' '
            # If it is the desired subtree, add the required leaf node in the subtree to the sentence trunk.
            elif (subtree_i.label() in target_tree_dict) and len(subtree_i.leaves())<=6:
                noun = extract_noun(str(subtree_i))
                if noun==None:
                    noun=further_extract_noun(str(subtree_i))
                    if noun==None:
                        noun=subtree_i.leaves()[0]
                out_result = out_result + noun + ' '
            # Otherwise, call the 'judgeNode' function on the subtree 'subtree_i'.
            else:
                out_result=judgeNode(str(subtree_i),out_result)
    return out_result

def extract_noun(noun_tree):
    """Extract first word of the desired part of speech in the subtree.
    Args:
        noun_tree: str. The input tree.
    return:
        word: str or None. The first word of the desired part of speech or None.
    """
    pending_noun_tree=Tree.fromstring(noun_tree)
    word=None
    for subtree_i in pending_noun_tree:
        if subtree_i.label()=='NN':
            word= subtree_i.leaves()[0]
            return word
        elif subtree_i.height()>2:
            word=extract_noun(str(subtree_i))
        if word!=None:
            return word

def further_extract_noun(noun_tree):
    """On a looser scale, look for the first word that appears in a variety of desired parts of speech.

    Args:
        noun_tree: str. The input tree.

    return:
        word: str or None. The first word of a variety of desired parts of speech or None.
    """
    pending_noun_tree=Tree.fromstring(noun_tree)
    word=None
    for subtree_i in pending_noun_tree:
        if len(subtree_i.label())>2 and subtree_i.label()[0:2]=='NN':
            word=subtree_i.leaves()[0]
            return word
        elif subtree_i.height()>2:
            word=further_extract_noun(str(subtree_i))
        if word!=None:
            return word

def get_required_layer(tree):
    """Extract required layer.
        Args:
            tree: str. The input tree.
        return:
            result_queue_list: list. Required layer.
    """
    # Define the parameters needed to find the desired layers.
    condtion = False
    front = 0
    last = 1
    queue_list = []  # The list is implemented as a queue.
    result_queue_list = []  # This list is used to store the layer found.
    width = 1
    queue_list.append(tree)
    # Find the required layers.
    while front != last:
        condtion = False
        for i in range(width):
            k = 0
            for j in Tree.fromstring(queue_list[front + i]).subtrees():
                k += 1
            if k == 1:
                condtion = True
        # If found, the previous layers of the required layers is stored to facilitate subsequent operations.
        if condtion:
            for i in range(width):
                result_queue_list.append(queue_list[front + i])
            break
        # Otherwise, continue to look for the next layer.
        else:
            width = 0
            while front != last:
                for i in range(len(Tree.fromstring(queue_list[front]))):
                    queue_list.append(str(Tree.fromstring(queue_list[front])[i]))
                    width += 1
                front += 1
            last += width
    return result_queue_list

def transfrom(result_queue_list):
    """Extract sentence trunk.
    Args:
        result_queue_list: list. The input required layer.
    return:
        out_result: str. Sentence trunk.
    """
    out_result = ''
    first=1
    except_index=0
    for subtree_index in range(len(result_queue_list)):
        # If its subtree is a leaf node,we add its leaf directly to our sentence trunk.
        if type(Tree.fromstring(result_queue_list[subtree_index])[0]) == str:
            out_result = out_result + str(Tree.fromstring(result_queue_list[subtree_index])[0]) + ' '
        # Otherwise, the subtree is processed.
        else:
            tlable=Tree.fromstring(result_queue_list[subtree_index])
            # For the qualified subject, add all its leaf nodes to the sentence trunk, that is, save the qualified subject completely.
            if(tlable.label() in target_tree_dict) and len(tlable.leaves())<=6 and first==1:
                out_result=out_result+' '.join(tlable.leaves())+' '
                first=0
            # If this is the required subtree and the number of leaf nodes of the subtree meets the requirements,
            # find the word of the target part of speech in the subtree, and add the first word to the sentence trunk after finding it.
            # If the word of the target part of speech does not appears in the subtree, add the first leaf node of the subtree to the sentence trunk by default.
            elif(tlable.label() in target_tree_dict) and len(tlable.leaves())<=6:
                noun=extract_noun(str(tlable))
                if noun==None:
                    noun=further_extract_noun(str(tlable))
                    if noun==None:
                        noun=tlable.leaves()[0]
                out_result=out_result+noun+' '
            # Otherwise, search through subtree of the subtree.
            else:
                for subtree_j in Tree.fromstring(result_queue_list[subtree_index]):
                    try:
                        if type(subtree_j[0]) == str:
                            out_result = out_result + subtree_j[0] + ' '
                        elif judge(str(subtree_j)):
                            twait=Tree.fromstring(str(subtree_j))
                            out_result = out_result + str(twait.leaves()[0]) + ' '
                        else:
                            out_result=judgeNode(str(subtree_j),out_result)
                    except:
                        except_index=1
                        continue
    out_result=out_result.rstrip()
    if except_index==1:
        out_result=out_result+'   <Exception>(An exception occurred in the analysis of this sentence. The trunk of this sentence is simply processed.)'
    return out_result

def dealBracket(i):# i is the sring (tree)  # deal with the exception caused by the bracket problem
    if ('(' in i or ')' in i) and ('<' not in i and '>' not in i):
        i = i.replace('(', '<')
        i = i.replace(')', '>')
        return 1, i
    elif ('(' in i or ')' in i) and ('[' not in i and ']' not in i):
        i = i.replace('(', '[')
        i = i.replace(')', ']')
        return 2, i
    elif ('(' in i or ')' in i) and ('{' not in i and '}' not in i):
        i = i.replace('(', '{')
        i = i.replace(')', '}')
        return 3, i
    return 0, i
        
def dealStringReplace(s, srcList, tgtList):
    for n,it in enumerate(srcList):
        s = s.replace(it, tgtList[n])
    return s

minSentenceLengh = 8
mylist = ['(',')']
list1 = ['<','>']
list2 = ['[',']']
list3 = ['{','}']

if __name__ == "__main__":
    # Load language processing tools
    language_processing_tool = StanfordCoreNLP('./stanford-corenlp-full-2018-01-31',lang='en')
    num=0
    # The first file 'train.en100.txt' is the raw data to be processed.
    # The sentence trunk of each sentence will be written to the second file 'text_en.txt'.
    inputFilename = sys.argv[1] 
    outFilename = "%s.trunk" % inputFilename
    with open(inputFilename,'r',encoding='utf-8') as fileRead, open(outFilename,'w',encoding='utf-8') as fileWrite:
        for line in fileRead:
             num+=1
             print('-'*10 + str(num)+'-'*10)
             try:
                 case, stringTree = dealBracket(line.strip())
                 sentence_parser = language_processing_tool.parse(stringTree)
                 sentence_tree = Tree.fromstring(sentence_parser)
                 leaves_list=sentence_tree.leaves()
                 sentence_trunk = None
                 if len(leaves_list)< minSentenceLengh:# For short sentences, they are written to the target file directly.
                     sentence_trunk=' '.join(leaves_list)
                 else:
                     required_layer=get_required_layer(sentence_parser)
                     sentence_trunk=transfrom(required_layer)
                 if case == 1:
                     sentence_trunk = dealStringReplace(sentence_trunk, list1, mylist)
                 elif case == 2:
                     sentence_trunk = dealStringReplace(sentence_trunk, list2, mylist)
                 elif case == 3:
                     sentence_trunk = dealStringReplace(sentence_trunk, list3, mylist)
             except:
                 pass
             fileWrite.write("%s\n" % sentence_trunk)
    language_processing_tool.close()

