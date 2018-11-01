// GaussNoiseDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "Task3.h"
#include "GaussNoiseDlg.h"
#include "afxdialogex.h"


// GaussNoiseDlg 对话框

IMPLEMENT_DYNAMIC(GaussNoiseDlg, CDialog)

GaussNoiseDlg::GaussNoiseDlg(CWnd* pParent /*=nullptr*/)
	: CDialog(IDD_GAUSS_NOISE, pParent)
{

}

GaussNoiseDlg::~GaussNoiseDlg()
{
}

void GaussNoiseDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_MEANS, mMeans);
	DDX_Control(pDX, IDC_VARIANCE, mVariance);
}


BEGIN_MESSAGE_MAP(GaussNoiseDlg, CDialog)
	//ON_EN_CHANGE(IDC_EDIT2, &GaussNoiseDlg::OnEnChangeEdit2)
END_MESSAGE_MAP()


// GaussNoiseDlg 消息处理程序


void GaussNoiseDlg::OnEnChangeEdit2()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialog::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}
