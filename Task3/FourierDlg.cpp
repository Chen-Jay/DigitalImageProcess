// FourierDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "Task3.h"
#include "FourierDlg.h"
#include "afxdialogex.h"


// FourierDlg 对话框

IMPLEMENT_DYNAMIC(FourierDlg, CDialog)

FourierDlg::FourierDlg(CWnd* pParent /*=nullptr*/)
	: CDialog(IDD_FOURIER, pParent)
{

}

FourierDlg::~FourierDlg()
{
}

void FourierDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(FourierDlg, CDialog)
END_MESSAGE_MAP()


// FourierDlg 消息处理程序
