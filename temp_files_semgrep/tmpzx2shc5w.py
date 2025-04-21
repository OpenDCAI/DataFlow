#include <iostream>
#include <queue>
#include <string>
using namespace std;

//template <class T>
//class Node {
//private:
//	Node<T> next_node;
//	T element;
//public:
//	Node(T n, Node<T> new_node);
//	Node<T> next();
//	T get_element();
//	void set_next(Node<T> new_node);
//};
//template <class T>
//Node<T>::Node(T n, Node<T> new_node) :element(n), next_node(new_node) {}
//
//template <class T>
//Node<T> Node<T>::next() {
//	return next_node;
//}
//
//template <class T>
//T Node<T>::get_element() {
//	return element;
//}
//
//template <class T>
//void Node<T>::set_next(Node<T> new_node) {
//	next_node = new_node;
//}
//
//template <class T>
//class Linked_list {
//private:
//	Node<T> head;
//	Node<T> tail;
//public:
//	Linked_list();
//	~Linked_list();
//	void push(T n);
//	void pop();
//	void clear();
//	T front();
//	T back();
//	bool empty();
//
//};
//template <class T>
//Linked_list<T>::Linked_list() {
//	head = NULL;
//	tail = NULL;
//}
//template <class T>
//Linked_list<T>::~Linked_list() {
//	clear();
//}
//template <class T>
//void Linked_list<T>::push(T n) {
//	if (empty()) {
//		head = new Node<T>(T n, NULL);
//		tail = head;
//		return;
//	}
//	Node<T> ptr = new Node<T>(T n, NULL);
//	tail->set_next(ptr);
//	tail = ptr;
//}
//template <class T>
//void Linked_list<T>::pop() {
//	if (empty())
//		return;
//	Node<T> ptr = head;
//	head = head->next();
//	delete ptr;
//}
//template <class T>
//void Linked_list<T>::clear() {
//	while (!empty())
//		pop();
//}
//template <class T>
//T Linked_list<T>::front() {
//	if (empty())
//		return;
//	return head->get_element();
//}
//template <class T>
//T Linked_list<T>::back() {
//	if (empty())
//		return;
//	return tail->get_element();
//}
//template<class T>
//bool Linked_list<T>::empty() {
//	return head == NULL;
//}

template <class T>
class Tree {
private:
	Tree<T>* parent;
//	Linked_list<Tree*> children;
	Tree<T>* left_child;
	Tree<T>* right_child;
	T element;

public:
	Tree();
	Tree(Tree<T>* new_parent, Tree<T>* new_left, Tree<T>* new_right, T n);
	~Tree();
	T get_element();
	void set_element(T n);
	Tree<T>* get_parent();
	void set_parent(Tree<T>* new_parent);
	Tree<T>* get_left();
	Tree<T>* get_right();
	void set_left(Tree* new_child);
	void set_right(Tree* new_child);
	void set_children(Tree* left_child, Tree* right_child);
	void clear();
	bool is_leaf();
	bool is_root();
	bool is_left();
	bool is_right();
};

template <class T>
Tree<T>::Tree() {
	parent = NULL;
	left_child = NULL;
	right_child = NULL;
}

template <class T>
Tree<T>::Tree(Tree<T>* new_parent, Tree<T>* new_left, Tree<T>* new_right, T n) :
	parent(new_parent),
	left_child(new_left),
	right_child(new_right),
	element(n)
{}
template <class T>
Tree<T>::~Tree() {
}
template <class T>
T Tree<T>::get_element() {
	return element;
}
template <class T>
void Tree<T>::set_element(T n) {
	element = n;
}
template <class T>
Tree<T>* Tree<T>::get_parent() {
	return parent;
}
template <class T>
void Tree<T>::set_parent(Tree<T>* new_parent) {
	parent = new_parent;
}
template <class T>
Tree<T>* Tree<T>::get_left() {
	return left_child;
}
template <class T>
Tree<T>* Tree<T>::get_right() {
	return right_child;
}
template <class T>
void Tree<T>::set_left(Tree* new_child) {
	left_child = new_child;
}
template <class T>
void Tree<T>::set_right(Tree* new_child) {
	right_child = new_child;
}
template <class T>
void Tree<T>::set_children(Tree* left_child, Tree* right_child) {
	this->left_child = left_child;
	this->right_child = right_child;
}
template <class T>
void Tree<T>::clear() {
	if (this->is_left())
		this->get_left()->clear();
	if (this->is_right())
		this->get_right()->clear();
	delete this;
	
}

template <class T>
bool Tree<T>::is_leaf() {
	return (left_child == NULL && right_child == NULL);
}

template <class T>
bool Tree<T>::is_root() {
	return (parent == NULL);
}
template <class T>
bool Tree<T>::is_left() {
	return left_child != NULL;
}
template <class T>
bool Tree<T>::is_right() {
	return right_child != NULL;
}
template <class T>
void preorder(Tree<T>* tree);
template <class T>
void inorder(Tree<T>* tree);
template <class T>
void postorder(Tree<T>* tree);
template <class T>
Tree<T>* BFS(Tree<T>* root, T n);
int main() {
	int N;
	char root, left, right;
	cin >> N;
	Tree<char>* tmp;
	Tree<char>* root_tree;
	
	/*if (left != '.' && right != '.') {
		root_tree = new Tree<char>(NULL, new Tree<char>(root_tree, NULL, NULL, left),
			new Tree<char>(root_tree, NULL, NULL, right), root);
		BFS.push(root_tree->get_left());
		BFS.push(root_tree->get_right());
	}
	else if (left == '.') {
		root_tree = new Tree<char>(NULL, NULL,
			new Tree<char>(root_tree, NULL, NULL, right), root);
		BFS.push(root_tree->get_right());
	}
	else if (right == '.') {
		root_tree = new Tree<char>(NULL, new Tree<char>(root_tree, NULL, NULL, left),
			NULL, root);
		BFS.push(root_tree->get_left());
	}*/

	for (int i = 0; i < N; i++) {
		cin >> root >> left >> right;
		if (i == 0) {
			root_tree = new Tree<char>(NULL, NULL, NULL, root);
			
		}
		tmp = BFS(root_tree, root);
		if (left == '.' && right != '.') 
			tmp->set_children(NULL, new Tree<char>(tmp, NULL, NULL, right));
		
		else if (left != '.' && right == '.') 
			tmp->set_children(new Tree<char>(tmp, NULL, NULL, left), NULL);
		
		else if (left != '.' && right != '.') 
			tmp->set_children(new Tree<char>(tmp, NULL, NULL, left), new Tree<char>(tmp, NULL, NULL, right));
		
		
	}
	preorder(root_tree);
	cout << '\n';
	inorder(root_tree);
	cout << '\n';
	postorder(root_tree);
	cout << '\n';


	root_tree->clear();
	return 0;

}

template <class T>
void preorder(Tree<T>* tree) {
	cout << tree->get_element();
	if (tree->is_left())
		preorder(tree->get_left());
	if(tree->is_right())
		preorder(tree->get_right());
}
template <class T>
void inorder(Tree<T>* tree) {
	if(tree->is_left())
		inorder(tree->get_left());
	cout << tree->get_element();
	if(tree->is_right())
		inorder(tree->get_right());
}
template <class T>
void postorder(Tree<T>* tree) {
	if (tree->is_left())
		postorder(tree->get_left());
	if (tree->is_right())
		postorder(tree->get_right());
	cout << tree->get_element();
}
template <class T>
Tree<T>* BFS(Tree<T>* root, T n) {
	queue<Tree<char>*> BFS_find;
	BFS_find.push(root);
	Tree<char>* ptr;
	while (1) {
		if (BFS_find.empty())
			return NULL;
		ptr = BFS_find.front();
		BFS_find.pop();
		if (ptr->get_element() == n)
			return ptr;
		
		if (ptr->is_left())
			BFS_find.push(ptr->get_left());
		if (ptr->is_right())
			BFS_find.push(ptr->get_right());
	}
}
