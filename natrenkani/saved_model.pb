??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
)coordinate_transform_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*:
shared_name+)coordinate_transform_network/dense/kernel
?
=coordinate_transform_network/dense/kernel/Read/ReadVariableOpReadVariableOp)coordinate_transform_network/dense/kernel*
_output_shapes
:	?*
dtype0
?
'coordinate_transform_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'coordinate_transform_network/dense/bias
?
;coordinate_transform_network/dense/bias/Read/ReadVariableOpReadVariableOp'coordinate_transform_network/dense/bias*
_output_shapes	
:?*
dtype0
?
+coordinate_transform_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*<
shared_name-+coordinate_transform_network/dense_1/kernel
?
?coordinate_transform_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp+coordinate_transform_network/dense_1/kernel*
_output_shapes
:	?@*
dtype0
?
)coordinate_transform_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)coordinate_transform_network/dense_1/bias
?
=coordinate_transform_network/dense_1/bias/Read/ReadVariableOpReadVariableOp)coordinate_transform_network/dense_1/bias*
_output_shapes
:@*
dtype0
?
+coordinate_transform_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*<
shared_name-+coordinate_transform_network/dense_2/kernel
?
?coordinate_transform_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp+coordinate_transform_network/dense_2/kernel*
_output_shapes

:@2*
dtype0
?
)coordinate_transform_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*:
shared_name+)coordinate_transform_network/dense_2/bias
?
=coordinate_transform_network/dense_2/bias/Read/ReadVariableOpReadVariableOp)coordinate_transform_network/dense_2/bias*
_output_shapes
:2*
dtype0
?
+coordinate_transform_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*<
shared_name-+coordinate_transform_network/dense_3/kernel
?
?coordinate_transform_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp+coordinate_transform_network/dense_3/kernel*
_output_shapes

:2@*
dtype0
?
)coordinate_transform_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)coordinate_transform_network/dense_3/bias
?
=coordinate_transform_network/dense_3/bias/Read/ReadVariableOpReadVariableOp)coordinate_transform_network/dense_3/bias*
_output_shapes
:@*
dtype0
?
+coordinate_transform_network/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*<
shared_name-+coordinate_transform_network/dense_4/kernel
?
?coordinate_transform_network/dense_4/kernel/Read/ReadVariableOpReadVariableOp+coordinate_transform_network/dense_4/kernel*
_output_shapes
:	@?*
dtype0
?
)coordinate_transform_network/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)coordinate_transform_network/dense_4/bias
?
=coordinate_transform_network/dense_4/bias/Read/ReadVariableOpReadVariableOp)coordinate_transform_network/dense_4/bias*
_output_shapes	
:?*
dtype0
?
+coordinate_transform_network/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*<
shared_name-+coordinate_transform_network/dense_5/kernel
?
?coordinate_transform_network/dense_5/kernel/Read/ReadVariableOpReadVariableOp+coordinate_transform_network/dense_5/kernel*
_output_shapes
:	?*
dtype0
?
)coordinate_transform_network/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)coordinate_transform_network/dense_5/bias
?
=coordinate_transform_network/dense_5/bias/Read/ReadVariableOpReadVariableOp)coordinate_transform_network/dense_5/bias*
_output_shapes
:*
dtype0
?
+coordinate_transform_network/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*<
shared_name-+coordinate_transform_network/dense_6/kernel
?
?coordinate_transform_network/dense_6/kernel/Read/ReadVariableOpReadVariableOp+coordinate_transform_network/dense_6/kernel*
_output_shapes

:22*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
0Adam/coordinate_transform_network/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*A
shared_name20Adam/coordinate_transform_network/dense/kernel/m
?
DAdam/coordinate_transform_network/dense/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense/kernel/m*
_output_shapes
:	?*
dtype0
?
.Adam/coordinate_transform_network/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/coordinate_transform_network/dense/bias/m
?
BAdam/coordinate_transform_network/dense/bias/m/Read/ReadVariableOpReadVariableOp.Adam/coordinate_transform_network/dense/bias/m*
_output_shapes	
:?*
dtype0
?
2Adam/coordinate_transform_network/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*C
shared_name42Adam/coordinate_transform_network/dense_1/kernel/m
?
FAdam/coordinate_transform_network/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_1/kernel/m*
_output_shapes
:	?@*
dtype0
?
0Adam/coordinate_transform_network/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/coordinate_transform_network/dense_1/bias/m
?
DAdam/coordinate_transform_network/dense_1/bias/m/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
2Adam/coordinate_transform_network/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*C
shared_name42Adam/coordinate_transform_network/dense_2/kernel/m
?
FAdam/coordinate_transform_network/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_2/kernel/m*
_output_shapes

:@2*
dtype0
?
0Adam/coordinate_transform_network/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*A
shared_name20Adam/coordinate_transform_network/dense_2/bias/m
?
DAdam/coordinate_transform_network/dense_2/bias/m/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_2/bias/m*
_output_shapes
:2*
dtype0
?
2Adam/coordinate_transform_network/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*C
shared_name42Adam/coordinate_transform_network/dense_3/kernel/m
?
FAdam/coordinate_transform_network/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_3/kernel/m*
_output_shapes

:2@*
dtype0
?
0Adam/coordinate_transform_network/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/coordinate_transform_network/dense_3/bias/m
?
DAdam/coordinate_transform_network/dense_3/bias/m/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_3/bias/m*
_output_shapes
:@*
dtype0
?
2Adam/coordinate_transform_network/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*C
shared_name42Adam/coordinate_transform_network/dense_4/kernel/m
?
FAdam/coordinate_transform_network/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_4/kernel/m*
_output_shapes
:	@?*
dtype0
?
0Adam/coordinate_transform_network/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/coordinate_transform_network/dense_4/bias/m
?
DAdam/coordinate_transform_network/dense_4/bias/m/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_4/bias/m*
_output_shapes	
:?*
dtype0
?
2Adam/coordinate_transform_network/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*C
shared_name42Adam/coordinate_transform_network/dense_5/kernel/m
?
FAdam/coordinate_transform_network/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_5/kernel/m*
_output_shapes
:	?*
dtype0
?
0Adam/coordinate_transform_network/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/coordinate_transform_network/dense_5/bias/m
?
DAdam/coordinate_transform_network/dense_5/bias/m/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_5/bias/m*
_output_shapes
:*
dtype0
?
2Adam/coordinate_transform_network/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*C
shared_name42Adam/coordinate_transform_network/dense_6/kernel/m
?
FAdam/coordinate_transform_network/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_6/kernel/m*
_output_shapes

:22*
dtype0
?
0Adam/coordinate_transform_network/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*A
shared_name20Adam/coordinate_transform_network/dense/kernel/v
?
DAdam/coordinate_transform_network/dense/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense/kernel/v*
_output_shapes
:	?*
dtype0
?
.Adam/coordinate_transform_network/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/coordinate_transform_network/dense/bias/v
?
BAdam/coordinate_transform_network/dense/bias/v/Read/ReadVariableOpReadVariableOp.Adam/coordinate_transform_network/dense/bias/v*
_output_shapes	
:?*
dtype0
?
2Adam/coordinate_transform_network/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*C
shared_name42Adam/coordinate_transform_network/dense_1/kernel/v
?
FAdam/coordinate_transform_network/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_1/kernel/v*
_output_shapes
:	?@*
dtype0
?
0Adam/coordinate_transform_network/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/coordinate_transform_network/dense_1/bias/v
?
DAdam/coordinate_transform_network/dense_1/bias/v/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
2Adam/coordinate_transform_network/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*C
shared_name42Adam/coordinate_transform_network/dense_2/kernel/v
?
FAdam/coordinate_transform_network/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_2/kernel/v*
_output_shapes

:@2*
dtype0
?
0Adam/coordinate_transform_network/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*A
shared_name20Adam/coordinate_transform_network/dense_2/bias/v
?
DAdam/coordinate_transform_network/dense_2/bias/v/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_2/bias/v*
_output_shapes
:2*
dtype0
?
2Adam/coordinate_transform_network/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2@*C
shared_name42Adam/coordinate_transform_network/dense_3/kernel/v
?
FAdam/coordinate_transform_network/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_3/kernel/v*
_output_shapes

:2@*
dtype0
?
0Adam/coordinate_transform_network/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/coordinate_transform_network/dense_3/bias/v
?
DAdam/coordinate_transform_network/dense_3/bias/v/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_3/bias/v*
_output_shapes
:@*
dtype0
?
2Adam/coordinate_transform_network/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*C
shared_name42Adam/coordinate_transform_network/dense_4/kernel/v
?
FAdam/coordinate_transform_network/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_4/kernel/v*
_output_shapes
:	@?*
dtype0
?
0Adam/coordinate_transform_network/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/coordinate_transform_network/dense_4/bias/v
?
DAdam/coordinate_transform_network/dense_4/bias/v/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_4/bias/v*
_output_shapes	
:?*
dtype0
?
2Adam/coordinate_transform_network/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*C
shared_name42Adam/coordinate_transform_network/dense_5/kernel/v
?
FAdam/coordinate_transform_network/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_5/kernel/v*
_output_shapes
:	?*
dtype0
?
0Adam/coordinate_transform_network/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/coordinate_transform_network/dense_5/bias/v
?
DAdam/coordinate_transform_network/dense_5/bias/v/Read/ReadVariableOpReadVariableOp0Adam/coordinate_transform_network/dense_5/bias/v*
_output_shapes
:*
dtype0
?
2Adam/coordinate_transform_network/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*C
shared_name42Adam/coordinate_transform_network/dense_6/kernel/v
?
FAdam/coordinate_transform_network/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/coordinate_transform_network/dense_6/kernel/v*
_output_shapes

:22*
dtype0

NoOpNoOp
?E
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?E
value?DB?D B?D
?
d1
d2
	embedding
r1
r2
restored
U
	optimizer
	loss

regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
^

3kernel
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratemjmkmlmmmnmo!mp"mq'mr(ms-mt.mu3mvvwvxvyvzv{v|!v}"v~'v(v?-v?.v?3v?
 
 
^
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
^
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312
?
=layer_regularization_losses

regularization_losses
trainable_variables
>layer_metrics
?non_trainable_variables
	variables

@layers
Ametrics
 
ca
VARIABLE_VALUE)coordinate_transform_network/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE'coordinate_transform_network/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Blayer_regularization_losses
regularization_losses
trainable_variables
Clayer_metrics
Dnon_trainable_variables
	variables

Elayers
Fmetrics
ec
VARIABLE_VALUE+coordinate_transform_network/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE)coordinate_transform_network/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Glayer_regularization_losses
regularization_losses
trainable_variables
Hlayer_metrics
Inon_trainable_variables
	variables

Jlayers
Kmetrics
lj
VARIABLE_VALUE+coordinate_transform_network/dense_2/kernel+embedding/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE)coordinate_transform_network/dense_2/bias)embedding/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Llayer_regularization_losses
regularization_losses
trainable_variables
Mlayer_metrics
Nnon_trainable_variables
	variables

Olayers
Pmetrics
ec
VARIABLE_VALUE+coordinate_transform_network/dense_3/kernel$r1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE)coordinate_transform_network/dense_3/bias"r1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
?
Qlayer_regularization_losses
#regularization_losses
$trainable_variables
Rlayer_metrics
Snon_trainable_variables
%	variables

Tlayers
Umetrics
ec
VARIABLE_VALUE+coordinate_transform_network/dense_4/kernel$r2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE)coordinate_transform_network/dense_4/bias"r2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
?
Vlayer_regularization_losses
)regularization_losses
*trainable_variables
Wlayer_metrics
Xnon_trainable_variables
+	variables

Ylayers
Zmetrics
ki
VARIABLE_VALUE+coordinate_transform_network/dense_5/kernel*restored/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE)coordinate_transform_network/dense_5/bias(restored/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
?
[layer_regularization_losses
/regularization_losses
0trainable_variables
\layer_metrics
]non_trainable_variables
1	variables

^layers
_metrics
db
VARIABLE_VALUE+coordinate_transform_network/dense_6/kernel#U/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

30

30
?
`layer_regularization_losses
4regularization_losses
5trainable_variables
alayer_metrics
bnon_trainable_variables
6	variables

clayers
dmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
1
0
1
2
3
4
5
6

e0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ftotal
	gcount
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

h	variables
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense/kernel/m@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/coordinate_transform_network/dense/bias/m>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_1/kernel/m@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_1/bias/m>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_2/kernel/mGembedding/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_2/bias/mEembedding/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_3/kernel/m@r1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_3/bias/m>r1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_4/kernel/m@r2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_4/bias/m>r2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_5/kernel/mFrestored/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_5/bias/mDrestored/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_6/kernel/m?U/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense/kernel/v@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/coordinate_transform_network/dense/bias/v>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_1/kernel/v@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_1/bias/v>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_2/kernel/vGembedding/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_2/bias/vEembedding/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_3/kernel/v@r1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_3/bias/v>r1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_4/kernel/v@r2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_4/bias/v>r2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_5/kernel/vFrestored/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/coordinate_transform_network/dense_5/bias/vDrestored/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/coordinate_transform_network/dense_6/kernel/v?U/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)coordinate_transform_network/dense/kernel'coordinate_transform_network/dense/bias+coordinate_transform_network/dense_1/kernel)coordinate_transform_network/dense_1/bias+coordinate_transform_network/dense_2/kernel)coordinate_transform_network/dense_2/bias+coordinate_transform_network/dense_6/kernel+coordinate_transform_network/dense_3/kernel)coordinate_transform_network/dense_3/bias+coordinate_transform_network/dense_4/kernel)coordinate_transform_network/dense_4/bias+coordinate_transform_network/dense_5/kernel)coordinate_transform_network/dense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_526321
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename=coordinate_transform_network/dense/kernel/Read/ReadVariableOp;coordinate_transform_network/dense/bias/Read/ReadVariableOp?coordinate_transform_network/dense_1/kernel/Read/ReadVariableOp=coordinate_transform_network/dense_1/bias/Read/ReadVariableOp?coordinate_transform_network/dense_2/kernel/Read/ReadVariableOp=coordinate_transform_network/dense_2/bias/Read/ReadVariableOp?coordinate_transform_network/dense_3/kernel/Read/ReadVariableOp=coordinate_transform_network/dense_3/bias/Read/ReadVariableOp?coordinate_transform_network/dense_4/kernel/Read/ReadVariableOp=coordinate_transform_network/dense_4/bias/Read/ReadVariableOp?coordinate_transform_network/dense_5/kernel/Read/ReadVariableOp=coordinate_transform_network/dense_5/bias/Read/ReadVariableOp?coordinate_transform_network/dense_6/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpDAdam/coordinate_transform_network/dense/kernel/m/Read/ReadVariableOpBAdam/coordinate_transform_network/dense/bias/m/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_1/kernel/m/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_1/bias/m/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_2/kernel/m/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_2/bias/m/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_3/kernel/m/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_3/bias/m/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_4/kernel/m/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_4/bias/m/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_5/kernel/m/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_5/bias/m/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_6/kernel/m/Read/ReadVariableOpDAdam/coordinate_transform_network/dense/kernel/v/Read/ReadVariableOpBAdam/coordinate_transform_network/dense/bias/v/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_1/kernel/v/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_1/bias/v/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_2/kernel/v/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_2/bias/v/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_3/kernel/v/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_3/bias/v/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_4/kernel/v/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_4/bias/v/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_5/kernel/v/Read/ReadVariableOpDAdam/coordinate_transform_network/dense_5/bias/v/Read/ReadVariableOpFAdam/coordinate_transform_network/dense_6/kernel/v/Read/ReadVariableOpConst*;
Tin4
220	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_526615
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)coordinate_transform_network/dense/kernel'coordinate_transform_network/dense/bias+coordinate_transform_network/dense_1/kernel)coordinate_transform_network/dense_1/bias+coordinate_transform_network/dense_2/kernel)coordinate_transform_network/dense_2/bias+coordinate_transform_network/dense_3/kernel)coordinate_transform_network/dense_3/bias+coordinate_transform_network/dense_4/kernel)coordinate_transform_network/dense_4/bias+coordinate_transform_network/dense_5/kernel)coordinate_transform_network/dense_5/bias+coordinate_transform_network/dense_6/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount0Adam/coordinate_transform_network/dense/kernel/m.Adam/coordinate_transform_network/dense/bias/m2Adam/coordinate_transform_network/dense_1/kernel/m0Adam/coordinate_transform_network/dense_1/bias/m2Adam/coordinate_transform_network/dense_2/kernel/m0Adam/coordinate_transform_network/dense_2/bias/m2Adam/coordinate_transform_network/dense_3/kernel/m0Adam/coordinate_transform_network/dense_3/bias/m2Adam/coordinate_transform_network/dense_4/kernel/m0Adam/coordinate_transform_network/dense_4/bias/m2Adam/coordinate_transform_network/dense_5/kernel/m0Adam/coordinate_transform_network/dense_5/bias/m2Adam/coordinate_transform_network/dense_6/kernel/m0Adam/coordinate_transform_network/dense/kernel/v.Adam/coordinate_transform_network/dense/bias/v2Adam/coordinate_transform_network/dense_1/kernel/v0Adam/coordinate_transform_network/dense_1/bias/v2Adam/coordinate_transform_network/dense_2/kernel/v0Adam/coordinate_transform_network/dense_2/bias/v2Adam/coordinate_transform_network/dense_3/kernel/v0Adam/coordinate_transform_network/dense_3/bias/v2Adam/coordinate_transform_network/dense_4/kernel/v0Adam/coordinate_transform_network/dense_4/bias/v2Adam/coordinate_transform_network/dense_5/kernel/v0Adam/coordinate_transform_network/dense_5/bias/v2Adam/coordinate_transform_network/dense_6/kernel/v*:
Tin3
12/*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_526763??
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_526352

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_5_layer_call_fn_526440

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5262312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_5_layer_call_and_return_conditional_losses_526431

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
X__inference_coordinate_transform_network_layer_call_and_return_conditional_losses_526248
input_1
dense_526089
dense_526091
dense_1_526116
dense_1_526118
dense_2_526143
dense_2_526145
dense_6_526164
dense_3_526189
dense_3_526191
dense_4_526216
dense_4_526218
dense_5_526242
dense_5_526244
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_526089dense_526091*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5260782
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_526116dense_1_526118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5261052!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_526143dense_2_526145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5261322!
dense_2/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_6_526164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5261552!
dense_6/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_3_526189dense_3_526191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5261782!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_526216dense_4_526218*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5262052!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_526242dense_5_526244*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5262312!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
C__inference_dense_6_layer_call_and_return_conditional_losses_526155

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
}
(__inference_dense_3_layer_call_fn_526401

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5261782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_526332

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_1_layer_call_fn_526361

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5261052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_dense_6_layer_call_and_return_conditional_losses_526447

inputs"
matmul_readvariableop_resource
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul|
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
C__inference_dense_5_layer_call_and_return_conditional_losses_526231

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_dense_layer_call_fn_526341

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5260782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_526132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
=__inference_coordinate_transform_network_layer_call_fn_526280
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_coordinate_transform_network_layer_call_and_return_conditional_losses_5262482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_526392

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_526105

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_526372

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
? 
"__inference__traced_restore_526763
file_prefix>
:assignvariableop_coordinate_transform_network_dense_kernel>
:assignvariableop_1_coordinate_transform_network_dense_biasB
>assignvariableop_2_coordinate_transform_network_dense_1_kernel@
<assignvariableop_3_coordinate_transform_network_dense_1_biasB
>assignvariableop_4_coordinate_transform_network_dense_2_kernel@
<assignvariableop_5_coordinate_transform_network_dense_2_biasB
>assignvariableop_6_coordinate_transform_network_dense_3_kernel@
<assignvariableop_7_coordinate_transform_network_dense_3_biasB
>assignvariableop_8_coordinate_transform_network_dense_4_kernel@
<assignvariableop_9_coordinate_transform_network_dense_4_biasC
?assignvariableop_10_coordinate_transform_network_dense_5_kernelA
=assignvariableop_11_coordinate_transform_network_dense_5_biasC
?assignvariableop_12_coordinate_transform_network_dense_6_kernel!
assignvariableop_13_adam_iter#
assignvariableop_14_adam_beta_1#
assignvariableop_15_adam_beta_2"
assignvariableop_16_adam_decay*
&assignvariableop_17_adam_learning_rate
assignvariableop_18_total
assignvariableop_19_countH
Dassignvariableop_20_adam_coordinate_transform_network_dense_kernel_mF
Bassignvariableop_21_adam_coordinate_transform_network_dense_bias_mJ
Fassignvariableop_22_adam_coordinate_transform_network_dense_1_kernel_mH
Dassignvariableop_23_adam_coordinate_transform_network_dense_1_bias_mJ
Fassignvariableop_24_adam_coordinate_transform_network_dense_2_kernel_mH
Dassignvariableop_25_adam_coordinate_transform_network_dense_2_bias_mJ
Fassignvariableop_26_adam_coordinate_transform_network_dense_3_kernel_mH
Dassignvariableop_27_adam_coordinate_transform_network_dense_3_bias_mJ
Fassignvariableop_28_adam_coordinate_transform_network_dense_4_kernel_mH
Dassignvariableop_29_adam_coordinate_transform_network_dense_4_bias_mJ
Fassignvariableop_30_adam_coordinate_transform_network_dense_5_kernel_mH
Dassignvariableop_31_adam_coordinate_transform_network_dense_5_bias_mJ
Fassignvariableop_32_adam_coordinate_transform_network_dense_6_kernel_mH
Dassignvariableop_33_adam_coordinate_transform_network_dense_kernel_vF
Bassignvariableop_34_adam_coordinate_transform_network_dense_bias_vJ
Fassignvariableop_35_adam_coordinate_transform_network_dense_1_kernel_vH
Dassignvariableop_36_adam_coordinate_transform_network_dense_1_bias_vJ
Fassignvariableop_37_adam_coordinate_transform_network_dense_2_kernel_vH
Dassignvariableop_38_adam_coordinate_transform_network_dense_2_bias_vJ
Fassignvariableop_39_adam_coordinate_transform_network_dense_3_kernel_vH
Dassignvariableop_40_adam_coordinate_transform_network_dense_3_bias_vJ
Fassignvariableop_41_adam_coordinate_transform_network_dense_4_kernel_vH
Dassignvariableop_42_adam_coordinate_transform_network_dense_4_bias_vJ
Fassignvariableop_43_adam_coordinate_transform_network_dense_5_kernel_vH
Dassignvariableop_44_adam_coordinate_transform_network_dense_5_bias_vJ
Fassignvariableop_45_adam_coordinate_transform_network_dense_6_kernel_v
identity_47??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB+embedding/kernel/.ATTRIBUTES/VARIABLE_VALUEB)embedding/bias/.ATTRIBUTES/VARIABLE_VALUEB$r1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"r1/bias/.ATTRIBUTES/VARIABLE_VALUEB$r2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"r2/bias/.ATTRIBUTES/VARIABLE_VALUEB*restored/kernel/.ATTRIBUTES/VARIABLE_VALUEB(restored/bias/.ATTRIBUTES/VARIABLE_VALUEB#U/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGembedding/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEembedding/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@r1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>r1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@r2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>r2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFrestored/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDrestored/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?U/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGembedding/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEembedding/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@r1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>r1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@r2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>r2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFrestored/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDrestored/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?U/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::*=
dtypes3
12/	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp:assignvariableop_coordinate_transform_network_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp:assignvariableop_1_coordinate_transform_network_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp>assignvariableop_2_coordinate_transform_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp<assignvariableop_3_coordinate_transform_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp>assignvariableop_4_coordinate_transform_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp<assignvariableop_5_coordinate_transform_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp>assignvariableop_6_coordinate_transform_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp<assignvariableop_7_coordinate_transform_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp>assignvariableop_8_coordinate_transform_network_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp<assignvariableop_9_coordinate_transform_network_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp?assignvariableop_10_coordinate_transform_network_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp=assignvariableop_11_coordinate_transform_network_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp?assignvariableop_12_coordinate_transform_network_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_adam_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpDassignvariableop_20_adam_coordinate_transform_network_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpBassignvariableop_21_adam_coordinate_transform_network_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpFassignvariableop_22_adam_coordinate_transform_network_dense_1_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpDassignvariableop_23_adam_coordinate_transform_network_dense_1_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpFassignvariableop_24_adam_coordinate_transform_network_dense_2_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpDassignvariableop_25_adam_coordinate_transform_network_dense_2_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpFassignvariableop_26_adam_coordinate_transform_network_dense_3_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpDassignvariableop_27_adam_coordinate_transform_network_dense_3_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpFassignvariableop_28_adam_coordinate_transform_network_dense_4_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpDassignvariableop_29_adam_coordinate_transform_network_dense_4_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpFassignvariableop_30_adam_coordinate_transform_network_dense_5_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpDassignvariableop_31_adam_coordinate_transform_network_dense_5_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpFassignvariableop_32_adam_coordinate_transform_network_dense_6_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpDassignvariableop_33_adam_coordinate_transform_network_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpBassignvariableop_34_adam_coordinate_transform_network_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpFassignvariableop_35_adam_coordinate_transform_network_dense_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpDassignvariableop_36_adam_coordinate_transform_network_dense_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpFassignvariableop_37_adam_coordinate_transform_network_dense_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpDassignvariableop_38_adam_coordinate_transform_network_dense_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpFassignvariableop_39_adam_coordinate_transform_network_dense_3_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpDassignvariableop_40_adam_coordinate_transform_network_dense_3_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpFassignvariableop_41_adam_coordinate_transform_network_dense_4_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpDassignvariableop_42_adam_coordinate_transform_network_dense_4_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpFassignvariableop_43_adam_coordinate_transform_network_dense_5_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpDassignvariableop_44_adam_coordinate_transform_network_dense_5_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpFassignvariableop_45_adam_coordinate_transform_network_dense_6_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_459
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_46Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_46?
Identity_47IdentityIdentity_46:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_47"#
identity_47Identity_47:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?k
?
__inference__traced_save_526615
file_prefixH
Dsavev2_coordinate_transform_network_dense_kernel_read_readvariableopF
Bsavev2_coordinate_transform_network_dense_bias_read_readvariableopJ
Fsavev2_coordinate_transform_network_dense_1_kernel_read_readvariableopH
Dsavev2_coordinate_transform_network_dense_1_bias_read_readvariableopJ
Fsavev2_coordinate_transform_network_dense_2_kernel_read_readvariableopH
Dsavev2_coordinate_transform_network_dense_2_bias_read_readvariableopJ
Fsavev2_coordinate_transform_network_dense_3_kernel_read_readvariableopH
Dsavev2_coordinate_transform_network_dense_3_bias_read_readvariableopJ
Fsavev2_coordinate_transform_network_dense_4_kernel_read_readvariableopH
Dsavev2_coordinate_transform_network_dense_4_bias_read_readvariableopJ
Fsavev2_coordinate_transform_network_dense_5_kernel_read_readvariableopH
Dsavev2_coordinate_transform_network_dense_5_bias_read_readvariableopJ
Fsavev2_coordinate_transform_network_dense_6_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_kernel_m_read_readvariableopM
Isavev2_adam_coordinate_transform_network_dense_bias_m_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_1_kernel_m_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_1_bias_m_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_2_kernel_m_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_2_bias_m_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_3_kernel_m_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_3_bias_m_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_4_kernel_m_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_4_bias_m_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_5_kernel_m_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_5_bias_m_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_6_kernel_m_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_kernel_v_read_readvariableopM
Isavev2_adam_coordinate_transform_network_dense_bias_v_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_1_kernel_v_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_1_bias_v_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_2_kernel_v_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_2_bias_v_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_3_kernel_v_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_3_bias_v_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_4_kernel_v_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_4_bias_v_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_5_kernel_v_read_readvariableopO
Ksavev2_adam_coordinate_transform_network_dense_5_bias_v_read_readvariableopQ
Msavev2_adam_coordinate_transform_network_dense_6_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*?
value?B?/B$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB+embedding/kernel/.ATTRIBUTES/VARIABLE_VALUEB)embedding/bias/.ATTRIBUTES/VARIABLE_VALUEB$r1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"r1/bias/.ATTRIBUTES/VARIABLE_VALUEB$r2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"r2/bias/.ATTRIBUTES/VARIABLE_VALUEB*restored/kernel/.ATTRIBUTES/VARIABLE_VALUEB(restored/bias/.ATTRIBUTES/VARIABLE_VALUEB#U/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGembedding/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEembedding/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@r1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>r1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@r2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>r2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFrestored/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDrestored/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?U/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGembedding/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEembedding/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@r1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>r1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@r2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>r2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFrestored/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDrestored/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?U/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:/*
dtype0*q
valuehBf/B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Dsavev2_coordinate_transform_network_dense_kernel_read_readvariableopBsavev2_coordinate_transform_network_dense_bias_read_readvariableopFsavev2_coordinate_transform_network_dense_1_kernel_read_readvariableopDsavev2_coordinate_transform_network_dense_1_bias_read_readvariableopFsavev2_coordinate_transform_network_dense_2_kernel_read_readvariableopDsavev2_coordinate_transform_network_dense_2_bias_read_readvariableopFsavev2_coordinate_transform_network_dense_3_kernel_read_readvariableopDsavev2_coordinate_transform_network_dense_3_bias_read_readvariableopFsavev2_coordinate_transform_network_dense_4_kernel_read_readvariableopDsavev2_coordinate_transform_network_dense_4_bias_read_readvariableopFsavev2_coordinate_transform_network_dense_5_kernel_read_readvariableopDsavev2_coordinate_transform_network_dense_5_bias_read_readvariableopFsavev2_coordinate_transform_network_dense_6_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_kernel_m_read_readvariableopIsavev2_adam_coordinate_transform_network_dense_bias_m_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_1_kernel_m_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_1_bias_m_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_2_kernel_m_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_2_bias_m_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_3_kernel_m_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_3_bias_m_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_4_kernel_m_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_4_bias_m_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_5_kernel_m_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_5_bias_m_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_6_kernel_m_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_kernel_v_read_readvariableopIsavev2_adam_coordinate_transform_network_dense_bias_v_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_1_kernel_v_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_1_bias_v_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_2_kernel_v_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_2_bias_v_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_3_kernel_v_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_3_bias_v_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_4_kernel_v_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_4_bias_v_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_5_kernel_v_read_readvariableopKsavev2_adam_coordinate_transform_network_dense_5_bias_v_read_readvariableopMsavev2_adam_coordinate_transform_network_dense_6_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *=
dtypes3
12/	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:?:	?@:@:@2:2:2@:@:	@?:?:	?::22: : : : : : : :	?:?:	?@:@:@2:2:2@:@:	@?:?:	?::22:	?:?:	?@:@:@2:2:2@:@:	@?:?:	?::22: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@2: 

_output_shapes
:2:$ 

_output_shapes

:2@: 

_output_shapes
:@:%	!

_output_shapes
:	@?:!


_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:22:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@2: 

_output_shapes
:2:$ 

_output_shapes

:2@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?:  

_output_shapes
::$! 

_output_shapes

:22:%"!

_output_shapes
:	?:!#

_output_shapes	
:?:%$!

_output_shapes
:	?@: %

_output_shapes
:@:$& 

_output_shapes

:@2: '

_output_shapes
:2:$( 

_output_shapes

:2@: )

_output_shapes
:@:%*!

_output_shapes
:	@?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?: -

_output_shapes
::$. 

_output_shapes

:22:/

_output_shapes
: 
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_526205

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_526078

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?a
?
!__inference__wrapped_model_526062
input_1E
Acoordinate_transform_network_dense_matmul_readvariableop_resourceF
Bcoordinate_transform_network_dense_biasadd_readvariableop_resourceG
Ccoordinate_transform_network_dense_1_matmul_readvariableop_resourceH
Dcoordinate_transform_network_dense_1_biasadd_readvariableop_resourceG
Ccoordinate_transform_network_dense_2_matmul_readvariableop_resourceH
Dcoordinate_transform_network_dense_2_biasadd_readvariableop_resourceG
Ccoordinate_transform_network_dense_6_matmul_readvariableop_resourceG
Ccoordinate_transform_network_dense_3_matmul_readvariableop_resourceH
Dcoordinate_transform_network_dense_3_biasadd_readvariableop_resourceG
Ccoordinate_transform_network_dense_4_matmul_readvariableop_resourceH
Dcoordinate_transform_network_dense_4_biasadd_readvariableop_resourceG
Ccoordinate_transform_network_dense_5_matmul_readvariableop_resourceH
Dcoordinate_transform_network_dense_5_biasadd_readvariableop_resource
identity??9coordinate_transform_network/dense/BiasAdd/ReadVariableOp?8coordinate_transform_network/dense/MatMul/ReadVariableOp?;coordinate_transform_network/dense_1/BiasAdd/ReadVariableOp?:coordinate_transform_network/dense_1/MatMul/ReadVariableOp?;coordinate_transform_network/dense_2/BiasAdd/ReadVariableOp?:coordinate_transform_network/dense_2/MatMul/ReadVariableOp?;coordinate_transform_network/dense_3/BiasAdd/ReadVariableOp?:coordinate_transform_network/dense_3/MatMul/ReadVariableOp?;coordinate_transform_network/dense_4/BiasAdd/ReadVariableOp?:coordinate_transform_network/dense_4/MatMul/ReadVariableOp?;coordinate_transform_network/dense_5/BiasAdd/ReadVariableOp?:coordinate_transform_network/dense_5/MatMul/ReadVariableOp?:coordinate_transform_network/dense_6/MatMul/ReadVariableOp?
8coordinate_transform_network/dense/MatMul/ReadVariableOpReadVariableOpAcoordinate_transform_network_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02:
8coordinate_transform_network/dense/MatMul/ReadVariableOp?
)coordinate_transform_network/dense/MatMulMatMulinput_1@coordinate_transform_network/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)coordinate_transform_network/dense/MatMul?
9coordinate_transform_network/dense/BiasAdd/ReadVariableOpReadVariableOpBcoordinate_transform_network_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9coordinate_transform_network/dense/BiasAdd/ReadVariableOp?
*coordinate_transform_network/dense/BiasAddBiasAdd3coordinate_transform_network/dense/MatMul:product:0Acoordinate_transform_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*coordinate_transform_network/dense/BiasAdd?
'coordinate_transform_network/dense/ReluRelu3coordinate_transform_network/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2)
'coordinate_transform_network/dense/Relu?
:coordinate_transform_network/dense_1/MatMul/ReadVariableOpReadVariableOpCcoordinate_transform_network_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02<
:coordinate_transform_network/dense_1/MatMul/ReadVariableOp?
+coordinate_transform_network/dense_1/MatMulMatMul5coordinate_transform_network/dense/Relu:activations:0Bcoordinate_transform_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2-
+coordinate_transform_network/dense_1/MatMul?
;coordinate_transform_network/dense_1/BiasAdd/ReadVariableOpReadVariableOpDcoordinate_transform_network_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;coordinate_transform_network/dense_1/BiasAdd/ReadVariableOp?
,coordinate_transform_network/dense_1/BiasAddBiasAdd5coordinate_transform_network/dense_1/MatMul:product:0Ccoordinate_transform_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2.
,coordinate_transform_network/dense_1/BiasAdd?
,coordinate_transform_network/dense_1/SigmoidSigmoid5coordinate_transform_network/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2.
,coordinate_transform_network/dense_1/Sigmoid?
:coordinate_transform_network/dense_2/MatMul/ReadVariableOpReadVariableOpCcoordinate_transform_network_dense_2_matmul_readvariableop_resource*
_output_shapes

:@2*
dtype02<
:coordinate_transform_network/dense_2/MatMul/ReadVariableOp?
+coordinate_transform_network/dense_2/MatMulMatMul0coordinate_transform_network/dense_1/Sigmoid:y:0Bcoordinate_transform_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+coordinate_transform_network/dense_2/MatMul?
;coordinate_transform_network/dense_2/BiasAdd/ReadVariableOpReadVariableOpDcoordinate_transform_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02=
;coordinate_transform_network/dense_2/BiasAdd/ReadVariableOp?
,coordinate_transform_network/dense_2/BiasAddBiasAdd5coordinate_transform_network/dense_2/MatMul:product:0Ccoordinate_transform_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22.
,coordinate_transform_network/dense_2/BiasAdd?
)coordinate_transform_network/dense_2/ReluRelu5coordinate_transform_network/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22+
)coordinate_transform_network/dense_2/Relu?
:coordinate_transform_network/dense_6/MatMul/ReadVariableOpReadVariableOpCcoordinate_transform_network_dense_6_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02<
:coordinate_transform_network/dense_6/MatMul/ReadVariableOp?
+coordinate_transform_network/dense_6/MatMulMatMul7coordinate_transform_network/dense_2/Relu:activations:0Bcoordinate_transform_network/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22-
+coordinate_transform_network/dense_6/MatMul?
:coordinate_transform_network/dense_3/MatMul/ReadVariableOpReadVariableOpCcoordinate_transform_network_dense_3_matmul_readvariableop_resource*
_output_shapes

:2@*
dtype02<
:coordinate_transform_network/dense_3/MatMul/ReadVariableOp?
+coordinate_transform_network/dense_3/MatMulMatMul5coordinate_transform_network/dense_6/MatMul:product:0Bcoordinate_transform_network/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2-
+coordinate_transform_network/dense_3/MatMul?
;coordinate_transform_network/dense_3/BiasAdd/ReadVariableOpReadVariableOpDcoordinate_transform_network_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;coordinate_transform_network/dense_3/BiasAdd/ReadVariableOp?
,coordinate_transform_network/dense_3/BiasAddBiasAdd5coordinate_transform_network/dense_3/MatMul:product:0Ccoordinate_transform_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2.
,coordinate_transform_network/dense_3/BiasAdd?
,coordinate_transform_network/dense_3/SigmoidSigmoid5coordinate_transform_network/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2.
,coordinate_transform_network/dense_3/Sigmoid?
:coordinate_transform_network/dense_4/MatMul/ReadVariableOpReadVariableOpCcoordinate_transform_network_dense_4_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02<
:coordinate_transform_network/dense_4/MatMul/ReadVariableOp?
+coordinate_transform_network/dense_4/MatMulMatMul0coordinate_transform_network/dense_3/Sigmoid:y:0Bcoordinate_transform_network/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+coordinate_transform_network/dense_4/MatMul?
;coordinate_transform_network/dense_4/BiasAdd/ReadVariableOpReadVariableOpDcoordinate_transform_network_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;coordinate_transform_network/dense_4/BiasAdd/ReadVariableOp?
,coordinate_transform_network/dense_4/BiasAddBiasAdd5coordinate_transform_network/dense_4/MatMul:product:0Ccoordinate_transform_network/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,coordinate_transform_network/dense_4/BiasAdd?
)coordinate_transform_network/dense_4/ReluRelu5coordinate_transform_network/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2+
)coordinate_transform_network/dense_4/Relu?
:coordinate_transform_network/dense_5/MatMul/ReadVariableOpReadVariableOpCcoordinate_transform_network_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02<
:coordinate_transform_network/dense_5/MatMul/ReadVariableOp?
+coordinate_transform_network/dense_5/MatMulMatMul7coordinate_transform_network/dense_4/Relu:activations:0Bcoordinate_transform_network/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+coordinate_transform_network/dense_5/MatMul?
;coordinate_transform_network/dense_5/BiasAdd/ReadVariableOpReadVariableOpDcoordinate_transform_network_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;coordinate_transform_network/dense_5/BiasAdd/ReadVariableOp?
,coordinate_transform_network/dense_5/BiasAddBiasAdd5coordinate_transform_network/dense_5/MatMul:product:0Ccoordinate_transform_network/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,coordinate_transform_network/dense_5/BiasAdd?
IdentityIdentity5coordinate_transform_network/dense_5/BiasAdd:output:0:^coordinate_transform_network/dense/BiasAdd/ReadVariableOp9^coordinate_transform_network/dense/MatMul/ReadVariableOp<^coordinate_transform_network/dense_1/BiasAdd/ReadVariableOp;^coordinate_transform_network/dense_1/MatMul/ReadVariableOp<^coordinate_transform_network/dense_2/BiasAdd/ReadVariableOp;^coordinate_transform_network/dense_2/MatMul/ReadVariableOp<^coordinate_transform_network/dense_3/BiasAdd/ReadVariableOp;^coordinate_transform_network/dense_3/MatMul/ReadVariableOp<^coordinate_transform_network/dense_4/BiasAdd/ReadVariableOp;^coordinate_transform_network/dense_4/MatMul/ReadVariableOp<^coordinate_transform_network/dense_5/BiasAdd/ReadVariableOp;^coordinate_transform_network/dense_5/MatMul/ReadVariableOp;^coordinate_transform_network/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::::2v
9coordinate_transform_network/dense/BiasAdd/ReadVariableOp9coordinate_transform_network/dense/BiasAdd/ReadVariableOp2t
8coordinate_transform_network/dense/MatMul/ReadVariableOp8coordinate_transform_network/dense/MatMul/ReadVariableOp2z
;coordinate_transform_network/dense_1/BiasAdd/ReadVariableOp;coordinate_transform_network/dense_1/BiasAdd/ReadVariableOp2x
:coordinate_transform_network/dense_1/MatMul/ReadVariableOp:coordinate_transform_network/dense_1/MatMul/ReadVariableOp2z
;coordinate_transform_network/dense_2/BiasAdd/ReadVariableOp;coordinate_transform_network/dense_2/BiasAdd/ReadVariableOp2x
:coordinate_transform_network/dense_2/MatMul/ReadVariableOp:coordinate_transform_network/dense_2/MatMul/ReadVariableOp2z
;coordinate_transform_network/dense_3/BiasAdd/ReadVariableOp;coordinate_transform_network/dense_3/BiasAdd/ReadVariableOp2x
:coordinate_transform_network/dense_3/MatMul/ReadVariableOp:coordinate_transform_network/dense_3/MatMul/ReadVariableOp2z
;coordinate_transform_network/dense_4/BiasAdd/ReadVariableOp;coordinate_transform_network/dense_4/BiasAdd/ReadVariableOp2x
:coordinate_transform_network/dense_4/MatMul/ReadVariableOp:coordinate_transform_network/dense_4/MatMul/ReadVariableOp2z
;coordinate_transform_network/dense_5/BiasAdd/ReadVariableOp;coordinate_transform_network/dense_5/BiasAdd/ReadVariableOp2x
:coordinate_transform_network/dense_5/MatMul/ReadVariableOp:coordinate_transform_network/dense_5/MatMul/ReadVariableOp2x
:coordinate_transform_network/dense_6/MatMul/ReadVariableOp:coordinate_transform_network/dense_6/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
n
(__inference_dense_6_layer_call_fn_526454

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5261552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
C__inference_dense_3_layer_call_and_return_conditional_losses_526178

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
}
(__inference_dense_4_layer_call_fn_526421

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5262052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_4_layer_call_and_return_conditional_losses_526412

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
(__inference_dense_2_layer_call_fn_526381

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5261322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_526321
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_5260622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
d1
d2
	embedding
r1
r2
restored
U
	optimizer
	loss

regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "CoordinateTransformNetwork", "name": "coordinate_transform_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CoordinateTransformNetwork"}, "training_config": {"loss": ["loss1", "loss2"], "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

3kernel
4regularization_losses
5trainable_variables
6	variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
8iter

9beta_1

:beta_2
	;decay
<learning_ratemjmkmlmmmnmo!mp"mq'mr(ms-mt.mu3mvvwvxvyvzv{v|!v}"v~'v(v?-v?.v?3v?"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312"
trackable_list_wrapper
~
0
1
2
3
4
5
!6
"7
'8
(9
-10
.11
312"
trackable_list_wrapper
?
=layer_regularization_losses

regularization_losses
trainable_variables
>layer_metrics
?non_trainable_variables
	variables

@layers
Ametrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
<::	?2)coordinate_transform_network/dense/kernel
6:4?2'coordinate_transform_network/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Blayer_regularization_losses
regularization_losses
trainable_variables
Clayer_metrics
Dnon_trainable_variables
	variables

Elayers
Fmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<	?@2+coordinate_transform_network/dense_1/kernel
7:5@2)coordinate_transform_network/dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Glayer_regularization_losses
regularization_losses
trainable_variables
Hlayer_metrics
Inon_trainable_variables
	variables

Jlayers
Kmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
=:;@22+coordinate_transform_network/dense_2/kernel
7:522)coordinate_transform_network/dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Llayer_regularization_losses
regularization_losses
trainable_variables
Mlayer_metrics
Nnon_trainable_variables
	variables

Olayers
Pmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
=:;2@2+coordinate_transform_network/dense_3/kernel
7:5@2)coordinate_transform_network/dense_3/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
Qlayer_regularization_losses
#regularization_losses
$trainable_variables
Rlayer_metrics
Snon_trainable_variables
%	variables

Tlayers
Umetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<	@?2+coordinate_transform_network/dense_4/kernel
8:6?2)coordinate_transform_network/dense_4/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
Vlayer_regularization_losses
)regularization_losses
*trainable_variables
Wlayer_metrics
Xnon_trainable_variables
+	variables

Ylayers
Zmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
>:<	?2+coordinate_transform_network/dense_5/kernel
7:52)coordinate_transform_network/dense_5/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
[layer_regularization_losses
/regularization_losses
0trainable_variables
\layer_metrics
]non_trainable_variables
1	variables

^layers
_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
=:;222+coordinate_transform_network/dense_6/kernel
 "
trackable_list_wrapper
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
?
`layer_regularization_losses
4regularization_losses
5trainable_variables
alayer_metrics
bnon_trainable_variables
6	variables

clayers
dmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
e0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	ftotal
	gcount
h	variables
i	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
f0
g1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
A:?	?20Adam/coordinate_transform_network/dense/kernel/m
;:9?2.Adam/coordinate_transform_network/dense/bias/m
C:A	?@22Adam/coordinate_transform_network/dense_1/kernel/m
<::@20Adam/coordinate_transform_network/dense_1/bias/m
B:@@222Adam/coordinate_transform_network/dense_2/kernel/m
<::220Adam/coordinate_transform_network/dense_2/bias/m
B:@2@22Adam/coordinate_transform_network/dense_3/kernel/m
<::@20Adam/coordinate_transform_network/dense_3/bias/m
C:A	@?22Adam/coordinate_transform_network/dense_4/kernel/m
=:;?20Adam/coordinate_transform_network/dense_4/bias/m
C:A	?22Adam/coordinate_transform_network/dense_5/kernel/m
<::20Adam/coordinate_transform_network/dense_5/bias/m
B:@2222Adam/coordinate_transform_network/dense_6/kernel/m
A:?	?20Adam/coordinate_transform_network/dense/kernel/v
;:9?2.Adam/coordinate_transform_network/dense/bias/v
C:A	?@22Adam/coordinate_transform_network/dense_1/kernel/v
<::@20Adam/coordinate_transform_network/dense_1/bias/v
B:@@222Adam/coordinate_transform_network/dense_2/kernel/v
<::220Adam/coordinate_transform_network/dense_2/bias/v
B:@2@22Adam/coordinate_transform_network/dense_3/kernel/v
<::@20Adam/coordinate_transform_network/dense_3/bias/v
C:A	@?22Adam/coordinate_transform_network/dense_4/kernel/v
=:;?20Adam/coordinate_transform_network/dense_4/bias/v
C:A	?22Adam/coordinate_transform_network/dense_5/kernel/v
<::20Adam/coordinate_transform_network/dense_5/bias/v
B:@2222Adam/coordinate_transform_network/dense_6/kernel/v
?2?
!__inference__wrapped_model_526062?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
=__inference_coordinate_transform_network_layer_call_fn_526280?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
X__inference_coordinate_transform_network_layer_call_and_return_conditional_losses_526248?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
&__inference_dense_layer_call_fn_526341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_526332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_526361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_526352?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_2_layer_call_fn_526381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_526372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_3_layer_call_fn_526401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_3_layer_call_and_return_conditional_losses_526392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_4_layer_call_fn_526421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_4_layer_call_and_return_conditional_losses_526412?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_5_layer_call_fn_526440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_5_layer_call_and_return_conditional_losses_526431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_6_layer_call_fn_526454?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_6_layer_call_and_return_conditional_losses_526447?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_526321input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_526062v3!"'(-.0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
X__inference_coordinate_transform_network_layer_call_and_return_conditional_losses_526248h3!"'(-.0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? ?
=__inference_coordinate_transform_network_layer_call_fn_526280[3!"'(-.0?-
&?#
!?
input_1?????????
? "???????????
C__inference_dense_1_layer_call_and_return_conditional_losses_526352]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
(__inference_dense_1_layer_call_fn_526361P0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_2_layer_call_and_return_conditional_losses_526372\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????2
? {
(__inference_dense_2_layer_call_fn_526381O/?,
%?"
 ?
inputs?????????@
? "??????????2?
C__inference_dense_3_layer_call_and_return_conditional_losses_526392\!"/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????@
? {
(__inference_dense_3_layer_call_fn_526401O!"/?,
%?"
 ?
inputs?????????2
? "??????????@?
C__inference_dense_4_layer_call_and_return_conditional_losses_526412]'(/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? |
(__inference_dense_4_layer_call_fn_526421P'(/?,
%?"
 ?
inputs?????????@
? "????????????
C__inference_dense_5_layer_call_and_return_conditional_losses_526431]-.0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_5_layer_call_fn_526440P-.0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_6_layer_call_and_return_conditional_losses_526447[3/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? z
(__inference_dense_6_layer_call_fn_526454N3/?,
%?"
 ?
inputs?????????2
? "??????????2?
A__inference_dense_layer_call_and_return_conditional_losses_526332]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? z
&__inference_dense_layer_call_fn_526341P/?,
%?"
 ?
inputs?????????
? "????????????
$__inference_signature_wrapper_526321?3!"'(-.;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????